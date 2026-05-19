#include <fr3_husky_controller/servers/fr3/sa_vive_tracker_action_server.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace fr3_husky_controller::servers::fr3
{

namespace
{
FR3ModelUpdater& getFR3ModelUpdater(ModelUpdaterBase& model_updater, const std::string& server_name)
{
    auto* fr3_model_updater = dynamic_cast<FR3ModelUpdater*>(&model_updater);
    if (!fr3_model_updater)
    {
        throw std::runtime_error("[" + server_name + "] requires FR3ModelUpdater");
    }
    return *fr3_model_updater;
}

// Extract robot name ("left" or "right") from an ee_name such as "left_fr3_hand_tcp".
// Returns an empty string if the prefix is not recognised.
std::string getRobotNameFromEEName(const std::string& ee_name)
{
    if (ee_name.rfind("left_", 0) == 0)  return "left";
    if (ee_name.rfind("right_", 0) == 0) return "right";
    return "";
}

Eigen::Vector3d extractSelectedAxis(const Eigen::Vector3d& selector)
{
    Eigen::Index dominant_idx = 0;
    const double dominant_value = selector.cwiseAbs().maxCoeff(&dominant_idx);
    if (dominant_value < 1e-6)
    {
        return Eigen::Vector3d::Zero();
    }

    Eigen::Vector3d axis = Eigen::Vector3d::Zero();
    axis(dominant_idx) = selector(dominant_idx) >= 0.0 ? 1.0 : -1.0;
    return axis;
}

Eigen::Vector3d projectVectorOntoAxis(const Eigen::Vector3d& vector, const Eigen::Vector3d& axis)
{
    const double axis_norm = axis.norm();
    if (axis_norm < 1e-9)
    {
        return vector;
    }

    const Eigen::Vector3d axis_unit = axis / axis_norm;
    return axis_unit * axis_unit.dot(vector);
}

Eigen::Matrix3d buildConstrainedOrientation(
    const Eigen::Matrix3d& locked_orientation,
    const Eigen::Matrix3d& raw_orientation,
    const Eigen::Vector3d& local_rotation_selector)
{
    const Eigen::Vector3d local_axis = extractSelectedAxis(local_rotation_selector);
    if (local_axis.norm() < 1e-6)
    {
        return locked_orientation;
    }

    const Eigen::Vector3d allowed_axis_world = locked_orientation * local_axis;

    const Eigen::Matrix3d relative_rotation_world =
        raw_orientation * locked_orientation.transpose();
    Eigen::AngleAxisd relative_aa(relative_rotation_world);

    if (std::abs(relative_aa.angle()) < 1e-9 || relative_aa.axis().squaredNorm() < 1e-9)
    {
        return locked_orientation;
    }

    const Eigen::Vector3d relative_rotvec_world =
        relative_aa.axis() * relative_aa.angle();
    const Eigen::Vector3d projected_rotvec_world =
        projectVectorOntoAxis(relative_rotvec_world, allowed_axis_world);
    const double projected_angle = projected_rotvec_world.norm();
    if (projected_angle < 1e-9)
    {
        return locked_orientation;
    }

    const Eigen::Matrix3d constrained_relative_world =
        Eigen::AngleAxisd(projected_angle, projected_rotvec_world / projected_angle).toRotationMatrix();
    return constrained_relative_world * locked_orientation;
}

sensor_msgs::msg::Image resizeImageNearest(
    const sensor_msgs::msg::Image& src,
    const uint32_t target_width,
    const uint32_t target_height)
{
    sensor_msgs::msg::Image dst;
    if (src.width == 0 || src.height == 0 || src.step == 0 || src.data.empty() ||
        target_width == 0 || target_height == 0)
    {
        return dst;
    }

    const uint32_t bytes_per_pixel = src.step / src.width;
    if (bytes_per_pixel == 0)
    {
        return dst;
    }

    dst = src;
    dst.width = target_width;
    dst.height = target_height;
    dst.step = target_width * bytes_per_pixel;
    dst.data.resize(static_cast<size_t>(dst.step) * dst.height);

    for (uint32_t y = 0; y < target_height; ++y)
    {
        const uint32_t src_y = (y * src.height) / target_height;
        for (uint32_t x = 0; x < target_width; ++x)
        {
            const uint32_t src_x = (x * src.width) / target_width;
            const size_t src_offset =
                static_cast<size_t>(src_y) * src.step + static_cast<size_t>(src_x) * bytes_per_pixel;
            const size_t dst_offset =
                static_cast<size_t>(y) * dst.step + static_cast<size_t>(x) * bytes_per_pixel;
            std::copy_n(src.data.begin() + static_cast<std::ptrdiff_t>(src_offset),
                        bytes_per_pixel,
                        dst.data.begin() + static_cast<std::ptrdiff_t>(dst_offset));
        }
    }

    return dst;
}

}  // namespace

SAViveTracker::SAViveTracker(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater)
: Base(name, node, model_updater),
  fr3_model_updater_(getFR3ModelUpdater(model_updater, name))
{
    pose_sub_         = node_->create_subscription<geometry_msgs::msg::PoseArray>("tracker_pose", 1, std::bind(&SAViveTracker::subPoseCallback, this, std::placeholders::_1));
    l_joy_sub_ = node_->create_subscription<sensor_msgs::msg::Joy>("lhand_joy", 1, std::bind(&SAViveTracker::subLJoyCallback, this, std::placeholders::_1));
    r_joy_sub_ = node_->create_subscription<sensor_msgs::msg::Joy>("rhand_joy", 1, std::bind(&SAViveTracker::subRJoyCallback, this, std::placeholders::_1));
    right_constraint_sub_ = node_->create_subscription<std_msgs::msg::Float64MultiArray>(
        "sa_right_eef_constraint",
        rclcpp::QoS(1).best_effort(),
        std::bind(&SAViveTracker::subRightConstraintCallback, this, std::placeholders::_1));
    front_overview_image_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
        "/mujoco_ros_hardware/front_overview/color/image_raw",
        rclcpp::SensorDataQoS(),
        std::bind(&SAViveTracker::subFrontOverviewImageCallback, this, std::placeholders::_1));
    front_overview_image_pub_ = node_->create_publisher<sensor_msgs::msg::Image>(
        "sa_front_overview/image_raw",
        rclcpp::QoS(1).best_effort());

    controller_poses_.assign(NUM_TRACKERS, Eigen::Affine3d::Identity());
    controller_poses_init_.assign(NUM_TRACKERS, Eigen::Affine3d::Identity());
    button_states_.assign(NUM_CONTROLLERS, std::vector<bool>(NUM_BUTTONS, false));
    prev_button_states_.assign(NUM_CONTROLLERS, std::vector<bool>(NUM_BUTTONS, false));

    tracker_base2robot_base_.assign(NUM_CONTROLLERS, Eigen::Matrix3d::Identity());

    ee_data_.clear();

    for(const auto& frame : fr3_model_updater_.robot_data_->getLinkFrameVector())
    {
        if (frame.find("wheel") != std::string::npos) 
        {
            has_mobile_ = true;
            break;
        }
    }

    // Action clients
    move_to_joint_client_ = rclcpp_action::create_client<MoveToJointAction>(node_, "fr3_move_to_joint");
    vt_self_client_       = rclcpp_action::create_client<ActionT>(node_, name_);

    // Subscribe to JTC action status to detect when trajectory execution completes
    auto jtc_qos = rclcpp::QoS(1).reliable().transient_local();
    jtc_status_sub_ = node_->create_subscription<action_msgs::msg::GoalStatusArray>(
        "fr3_joint_trajectory_controller/_action/status",
        jtc_qos,
        [this](const action_msgs::msg::GoalStatusArray::SharedPtr msg)
        {
            if (!waiting_for_jtc_.load(std::memory_order_relaxed)) return;

            bool jtc_busy = false;
            for (const auto& gs : msg->status_list)
            {
                if (gs.status == action_msgs::msg::GoalStatus::STATUS_EXECUTING ||
                    gs.status == action_msgs::msg::GoalStatus::STATUS_ACCEPTED)
                {
                    jtc_busy = true;
                    break;
                }
            }

            if (!jtc_busy)
            {
                waiting_for_jtc_.store(false, std::memory_order_relaxed);
                RCLCPP_INFO(node_->get_logger(),
                            "[%s] JTC finished — re-activating ViveTracker", name_.c_str());
                vt_self_client_->async_send_goal(saved_vive_goal_);
            }
        });

    // Initialize franka hand state
    for(const auto& robot_name : model_updater_.robot_names_) fr3_model_updater_.GripperHoming(robot_name); 

    RCLCPP_INFO(node_->get_logger(), "[%s] ViveTracker created", name_.c_str());
}

bool SAViveTracker::acceptGoal(const ActionT::Goal& goal)
{
    if (!model_updater_.HasEffortCommandInterface())
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: effort command interface is required",
                                         name_.c_str());
        return false;
    }

    if (goal.mode < 0 || goal.mode > 3)
    {
        RCLCPP_WARN(node_->get_logger(),
                                         "[%s] Reject action: mode must be 0 to 3 (0: CLIK, 1: OSF, 2: QPIK, 3: QPID). The mode from action goal is %d.",
                                         name_.c_str(),
                                         static_cast<int>(goal.mode));
        return false;
    }

    if(!goal.left_controller_ee_name.empty() && !fr3_model_updater_.robot_data_->hasLinkFrame(goal.left_controller_ee_name))
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: left_controller_ee_name from the goal [%s] is not includede in URDF.",
                                         name_.c_str(), goal.left_controller_ee_name.c_str());
        return false;
    }

    if(!goal.right_controller_ee_name.empty() && !fr3_model_updater_.robot_data_->hasLinkFrame(goal.right_controller_ee_name))
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: right_controller_ee_name from the goal [%s] is not includede in URDF.",
                                         name_.c_str(), goal.right_controller_ee_name.c_str());
        return false;
    }

    return true;
}

void SAViveTracker::onGoalAccepted(const ActionT::Goal& goal)
{
    control_mode_ = goal.mode;
    left_controller_ee_name_ = goal.left_controller_ee_name;
    right_controller_ee_name_ = goal.right_controller_ee_name;
    move_ori_ = goal.move_orientation;
    controller_pos_multiplier_ = static_cast<double>(goal.controller_pos_multiplier);
    controller_ori_multiplier_ = static_cast<double>(goal.controller_ori_multiplier);
    saved_vive_goal_ = goal;
    requestActivate();
}

void SAViveTracker::onStart()
{
    {
        std::lock_guard<std::mutex> lock(tracker_pose_mutex_);
        for(auto& tracker_pose : controller_poses_) tracker_pose.setIdentity();
    }
    for(auto& controller_pose_init : controller_poses_init_) controller_pose_init.setIdentity();
    {
        std::lock_guard<std::mutex> lock(button_state_mutex_);
        for(auto& button_state : button_states_) button_state = std::vector<bool>(NUM_BUTTONS, false);
    }

    for(auto& prev_button_state : prev_button_states_) prev_button_state = std::vector<bool>(NUM_BUTTONS, false);
    is_mouse_mode_on_.assign(NUM_CONTROLLERS, false);
    is_initialize_mode_on_ = false;
    is_gripper_mode_on_.assign(NUM_CONTROLLERS, false);
    ee_data_.clear();
    waiting_for_jtc_.store(false, std::memory_order_relaxed);
    right_constraint_orientation_locked_ = false;
    right_constraint_anchor_pose_locked_ = false;
    front_overview_publish_enabled_.store(false, std::memory_order_relaxed);
    front_overview_publish_log_pending_.store(false, std::memory_order_relaxed);
    last_front_overview_publish_time_ns_ = 0;
    front_overview_publish_until_ns_ = 0;

    if(!left_controller_ee_name_.empty())
    {
        ee_data_[left_controller_ee_name_] = drc::TaskSpaceData::Zero();
        ee_data_[left_controller_ee_name_].x = fr3_model_updater_.robot_data_->getPose(left_controller_ee_name_);
        ee_data_[left_controller_ee_name_].xdot = fr3_model_updater_.robot_data_->getVelocity(left_controller_ee_name_);
        ee_data_[left_controller_ee_name_].xddot.setZero();
        ee_data_[left_controller_ee_name_].setInit();
        ee_data_[left_controller_ee_name_].setDesired();
    }
    if(!right_controller_ee_name_.empty())
    {
        ee_data_[right_controller_ee_name_] = drc::TaskSpaceData::Zero();
        ee_data_[right_controller_ee_name_].x = fr3_model_updater_.robot_data_->getPose(right_controller_ee_name_);
        ee_data_[right_controller_ee_name_].xdot = fr3_model_updater_.robot_data_->getVelocity(right_controller_ee_name_);
        ee_data_[right_controller_ee_name_].xddot.setZero();
        ee_data_[right_controller_ee_name_].setInit();
        ee_data_[right_controller_ee_name_].setDesired();
    }

    RCLCPP_INFO(node_->get_logger(), "[%s] started", name_.c_str());
}

SAViveTracker::ComputeResult SAViveTracker::compute(const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/)
{
    for(auto& [ee_name, ee_data] : ee_data_)
    {
        ee_data.x = fr3_model_updater_.robot_data_->getPose(ee_name);
        ee_data.xdot = fr3_model_updater_.robot_data_->getVelocity(ee_name);
        ee_data.xddot.setZero();
    }

    std::vector<Eigen::Affine3d> controller_poses_local;   // left, right, head
    std::vector<std::vector<bool>> button_states_local;    // [left, right][trigger, grip, a, b]
    {
        std::lock_guard<std::mutex> lock(tracker_pose_mutex_);
        controller_poses_local = controller_poses_;
    }
    {
        std::lock_guard<std::mutex> lock(button_state_mutex_);
        button_states_local = button_states_;
    }
    Eigen::Vector3d right_constraint_vector_local = Eigen::Vector3d::Zero();
    bool right_constraint_received_local = false;
    bool right_constraint_applying_local = false;
    {
        std::lock_guard<std::mutex> lock(right_constraint_mutex_);
        right_constraint_vector_local = right_constraint_vector_;
        right_constraint_received_local = right_constraint_received_;
        right_constraint_applying_local = right_constraint_applying_;
    }

    // Initialize mode
    {
        if (!is_initialize_mode_on_)
        {
            // if "a" button on the vive controller is pressed
            if ((!prev_button_states_[IDX_LEFT_CON][IDX_A_BUTTON]  && button_states_local[IDX_LEFT_CON][IDX_A_BUTTON]) ||
                (!prev_button_states_[IDX_RIGHT_CON][IDX_A_BUTTON] && button_states_local[IDX_RIGHT_CON][IDX_A_BUTTON]))
            {
                is_initialize_mode_on_ = true;

                MoveToJointAction::Goal mtj_goal;
                for (const auto& robot_name : model_updater_.robot_names_)
                {
                    for (size_t j = 0; j < FR3_DOF; ++j)
                    {
                        mtj_goal.joint_names.push_back(robot_name + "_" + model_updater_.arm_id_ + "_joint" + std::to_string(j+1));
                        mtj_goal.target_positions.push_back(HomePose(j));
                        if(model_updater_.num_robots_ == 2 && has_mobile_ &&  j == 0)
                        {
                            if(robot_name.find("left")  != std::string::npos) mtj_goal.target_positions.back() -= M_PI / 6;
                            if(robot_name.find("right") != std::string::npos) mtj_goal.target_positions.back() += M_PI / 6;
                        }
                    }
                }
                mtj_goal.max_velocity_scaling_factor     = 0.1;
                mtj_goal.max_acceleration_scaling_factor = 0.1;

                // When MoveToJoint succeeds (trajectory sent to JTC), set waiting_for_jtc_ so
                // the JTC status subscriber re-activates ViveTracker after the robot finishes moving.
                auto send_opts = rclcpp_action::Client<MoveToJointAction>::SendGoalOptions();
                send_opts.result_callback =
                    [this](const rclcpp_action::ClientGoalHandle<MoveToJointAction>::WrappedResult&)
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] MoveToJoint done — waiting for JTC to finish", name_.c_str());
                        waiting_for_jtc_.store(true, std::memory_order_relaxed);
                    };

                move_to_joint_client_->async_send_goal(mtj_goal, send_opts);
                RCLCPP_INFO(node_->get_logger(),
                            "[%s] Initialize mode ON — goal sent to fr3_move_to_joint, yielding",
                            name_.c_str());

                // Yield: deactivate ViveTracker so MoveToJoint can become active_server_
                return ComputeResult::SUCCEEDED;
            }
        }
    }

    // Gripper control
    {
        if ((!prev_button_states_[IDX_LEFT_CON][IDX_B_BUTTON]  && button_states_local[IDX_LEFT_CON][IDX_B_BUTTON]) ||
            (!prev_button_states_[IDX_RIGHT_CON][IDX_B_BUTTON] && button_states_local[IDX_RIGHT_CON][IDX_B_BUTTON]))
        {
            const int64_t now_ns = node_->now().nanoseconds();
            front_overview_publish_enabled_.store(true, std::memory_order_relaxed);
            front_overview_publish_log_pending_.store(true, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lock(front_overview_publish_mutex_);
                last_front_overview_publish_time_ns_ = 0;
                front_overview_publish_until_ns_ = now_ns + 2000000000LL;
            }
            RCLCPP_INFO(node_->get_logger(),
                        "[%s] B button pressed -> publishing resized front_overview images for 2 seconds",
                        name_.c_str());
        }

        // Left controller
        if (!left_controller_ee_name_.empty())
        {
        if (!prev_button_states_[IDX_LEFT_CON][IDX_TRIGGER_BUTTON] && button_states_local[IDX_LEFT_CON][IDX_TRIGGER_BUTTON])
            {
                const std::string robot_name = getRobotNameFromEEName(left_controller_ee_name_);
                if (!robot_name.empty())
                {
                    is_gripper_mode_on_[IDX_LEFT_CON] = !is_gripper_mode_on_[IDX_LEFT_CON];
                    if (is_gripper_mode_on_[IDX_LEFT_CON])
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] lhand trigger released → GripperGrasp('%s')", name_.c_str(), robot_name.c_str());
                        fr3_model_updater_.GripperGrasp(robot_name);
                    }
                    else
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] lhand trigger released → GripperOpen('%s')", name_.c_str(), robot_name.c_str());
                        fr3_model_updater_.GripperOpen(robot_name);
                    }
                }
            }
        }

        // Right controller
        if (!right_controller_ee_name_.empty())
        {
            if (!prev_button_states_[IDX_RIGHT_CON][IDX_TRIGGER_BUTTON] && button_states_local[IDX_RIGHT_CON][IDX_TRIGGER_BUTTON])
            {
                const std::string robot_name = getRobotNameFromEEName(right_controller_ee_name_);
                if (!robot_name.empty())
                {
                    is_gripper_mode_on_[IDX_RIGHT_CON] = !is_gripper_mode_on_[IDX_RIGHT_CON];
                    if (is_gripper_mode_on_[IDX_RIGHT_CON])
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] rhand trigger released → GripperGrasp('%s')", name_.c_str(), robot_name.c_str());
                        fr3_model_updater_.GripperGrasp(robot_name);
                    }
                    else
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] rhand trigger released → GripperOpen('%s')", name_.c_str(), robot_name.c_str());
                        fr3_model_updater_.GripperOpen(robot_name);
                    }
                }
            }
        }
    }

    // Manipulator control
    {
        // Check mouse mode
        for(size_t i = 0; i < NUM_CONTROLLERS; ++i)
        {
            if(!is_mouse_mode_on_[i] && button_states_local[i][IDX_GRIP_BUTTON]) // activate mouse mode when "grip" button pressed
            {
                RCLCPP_INFO(node_->get_logger(), "[%s] %s Mouse Mode activated!", name_.c_str(), (i==0)?"Left":"Right");
                is_mouse_mode_on_[i] = true;
    
                controller_poses_init_[i] = controller_poses_local[i];
                if(i == 0 && !left_controller_ee_name_.empty())       ee_data_[left_controller_ee_name_].setInit();
                else if(i == 1 && !right_controller_ee_name_.empty()) ee_data_[right_controller_ee_name_].setInit();
            }
            else if(is_mouse_mode_on_[i] && !button_states_local[i][IDX_GRIP_BUTTON]) // deactivate mouse mode when grip button released
            {
                RCLCPP_INFO(node_->get_logger(), "[%s] %s Mouse Mode deactivated!", name_.c_str(), (i==0)?"Left":"Right");
                is_mouse_mode_on_[i] = false;
    
                controller_poses_init_[i] = controller_poses_local[i];
                if(i == IDX_LEFT_CON && !left_controller_ee_name_.empty())        ee_data_[left_controller_ee_name_].setInit();
                else if(i == IDX_RIGHT_CON && !right_controller_ee_name_.empty()) ee_data_[right_controller_ee_name_].setInit();
            }
        }
    
        
        if(!left_controller_ee_name_.empty()) // left vive controller
        {
            Eigen::Affine3d target_pose_diff; // EE init -> EE desired
            Eigen::Vector6d target_vel;
            target_pose_diff.setIdentity();
            target_vel.setZero();
            if(is_mouse_mode_on_[IDX_LEFT_CON])
            {
                const Eigen::Affine3d T_con_init2con_cur = controller_poses_init_[IDX_LEFT_CON].inverse() * controller_poses_local[IDX_LEFT_CON];
                const Eigen::Matrix3d R_ee_init2con_init = ee_data_[left_controller_ee_name_].x_init.linear().transpose() * tracker_base2robot_base_[IDX_LEFT_CON].transpose() * controller_poses_init_[IDX_LEFT_CON].linear();
    
                // Position
                target_pose_diff.translation() = controller_pos_multiplier_ * R_ee_init2con_init * T_con_init2con_cur.translation();
    
                // Orientation: using similarity transformation
                target_pose_diff.linear().setIdentity();
                if (move_ori_)
                {
                    const Eigen::AngleAxisd aa(T_con_init2con_cur.linear());
                    const Eigen::Matrix3d R_con_diff_scaled =
                        (std::abs(aa.angle()) > 1e-10)
                        ? Eigen::AngleAxisd(controller_ori_multiplier_ * aa.angle(), aa.axis()).toRotationMatrix()
                        : Eigen::Matrix3d::Identity();
    
                    target_pose_diff.linear() = R_ee_init2con_init * R_con_diff_scaled * R_ee_init2con_init.transpose();
                }
            }
    
            ee_data_[left_controller_ee_name_].x_desired = ee_data_[left_controller_ee_name_].x_init * target_pose_diff;
            ee_data_[left_controller_ee_name_].xdot_desired  = target_vel;
        }
    
        if(!right_controller_ee_name_.empty()) // right vive controller
        {
            Eigen::Affine3d target_pose_diff; // EE init -> EE desired
            Eigen::Vector6d target_vel;
            target_pose_diff.setIdentity();
            target_vel.setZero();
            if(is_mouse_mode_on_[IDX_RIGHT_CON])
            {    
                const Eigen::Affine3d T_con_init2con_cur = controller_poses_init_[IDX_RIGHT_CON].inverse() * controller_poses_local[IDX_RIGHT_CON];
                const Eigen::Matrix3d R_ee_init2con_init = ee_data_[right_controller_ee_name_].x_init.linear().transpose() * tracker_base2robot_base_[IDX_RIGHT_CON].transpose() * controller_poses_init_[IDX_RIGHT_CON].linear();
    
                // Position
                target_pose_diff.translation() = controller_pos_multiplier_ * R_ee_init2con_init * T_con_init2con_cur.translation();
    
                // Orientation: using similarity transformation
                target_pose_diff.linear().setIdentity();
                if (move_ori_)
                {
                    const Eigen::AngleAxisd aa(T_con_init2con_cur.linear());
                    const Eigen::Matrix3d R_con_diff_scaled =
                        (std::abs(aa.angle()) > 1e-10)
                        ? Eigen::AngleAxisd(controller_ori_multiplier_ * aa.angle(), aa.axis()).toRotationMatrix()
                        : Eigen::Matrix3d::Identity();
    
                    target_pose_diff.linear() = R_ee_init2con_init * R_con_diff_scaled * R_ee_init2con_init.transpose();
                }
            }

            Eigen::Affine3d raw_target = ee_data_[right_controller_ee_name_].x_init * target_pose_diff;
            if (right_constraint_received_local && right_constraint_applying_local)
            {
                Eigen::Matrix3d locked_orientation = Eigen::Matrix3d::Identity();
                Eigen::Affine3d anchor_pose = Eigen::Affine3d::Identity();
                {
                    std::lock_guard<std::mutex> lock(right_constraint_mutex_);
                    if (!right_constraint_orientation_locked_)
                    {
                        right_constraint_locked_orientation_ = ee_data_[right_controller_ee_name_].x.linear();
                        right_constraint_orientation_locked_ = true;
                    }
                    if (!right_constraint_anchor_pose_locked_)
                    {
                        right_constraint_anchor_pose_ = ee_data_[right_controller_ee_name_].x;
                        right_constraint_anchor_pose_locked_ = true;
                    }
                    locked_orientation = right_constraint_locked_orientation_;
                    anchor_pose = right_constraint_anchor_pose_;
                }

                const Eigen::Vector3d translation_selector = right_constraint_vector_local;
                const Eigen::Vector3d local_translation_axis =
                    extractSelectedAxis(translation_selector);
                if (local_translation_axis.norm() > 1e-6)
                {
                    constexpr double kOffAxisTranslationScale = 0.1;

                    const Eigen::Vector3d allowed_axis_world =
                        locked_orientation * local_translation_axis;

                    const Eigen::Vector3d delta_world_raw =
                        raw_target.translation() - anchor_pose.translation();
                    const Eigen::Vector3d delta_world_on_axis =
                        projectVectorOntoAxis(delta_world_raw, allowed_axis_world);
                    const Eigen::Vector3d delta_world_off_axis =
                        delta_world_raw - delta_world_on_axis;
                    const Eigen::Vector3d delta_world_regulated =
                        delta_world_on_axis + kOffAxisTranslationScale * delta_world_off_axis;
                    raw_target.translation() =
                        anchor_pose.translation() + delta_world_regulated;
                }

                raw_target.linear() = buildConstrainedOrientation(
                    locked_orientation,
                    raw_target.linear(),
                    right_constraint_vector_local);
            }

            ee_data_[right_controller_ee_name_].x_desired = raw_target;
            ee_data_[right_controller_ee_name_].xdot_desired  = target_vel;
        }
    
        bool is_qp_solved = true;
        std::string time_verbose = "";
        switch (control_mode_)
        {
            case 0: // CLIK
                fr3_model_updater_.robot_controller_->CLIKStep(ee_data_, fr3_model_updater_.qdot_desired_total_);
                fr3_model_updater_.q_desired_total_ = fr3_model_updater_.q_total_ +
                                                      fr3_model_updater_.dt_ * fr3_model_updater_.qdot_desired_total_;
                
                fr3_model_updater_.torque_desired_total_ = fr3_model_updater_.robot_controller_->moveJointTorqueStep(fr3_model_updater_.q_desired_total_,
                                                                                                                     fr3_model_updater_.qdot_desired_total_,
                                                                                                                     false);
                break;
            case 1: // OSF
                fr3_model_updater_.robot_controller_->OSFStep(ee_data_, fr3_model_updater_.torque_desired_total_);
                break;
            case 2: // QPIK
                is_qp_solved = fr3_model_updater_.robot_controller_->QPIKStep(ee_data_, fr3_model_updater_.qdot_desired_total_, time_verbose);
                if(!is_qp_solved) fr3_model_updater_.qdot_desired_total_.setZero();
                fr3_model_updater_.q_desired_total_ = fr3_model_updater_.q_total_ +
                                                      fr3_model_updater_.dt_ * fr3_model_updater_.qdot_desired_total_;
                fr3_model_updater_.torque_desired_total_ = fr3_model_updater_.robot_controller_->moveJointTorqueStep(fr3_model_updater_.q_desired_total_,
                                                                                                                     fr3_model_updater_.qdot_desired_total_,
                                                                                                                     false);
                break;
            case 3: // QPID
                is_qp_solved = fr3_model_updater_.robot_controller_->QPIDStep(ee_data_, fr3_model_updater_.torque_desired_total_, time_verbose);
                if(!is_qp_solved) fr3_model_updater_.torque_desired_total_ = fr3_model_updater_.robot_data_->getGravity();
                break;
            default:
                break;
        }
    
        fr3_model_updater_.writeCommand(fr3_model_updater_.torque_desired_total_ - fr3_model_updater_.g_total_); // robot_controller automatically add gravity force
    
        auto fb = std::make_shared<ActionT::Feedback>();
        fb->is_qp_solved = is_qp_solved;
        fb->time_verbose = time_verbose;
        publishFeedback(fb);
        
        prev_button_states_ = button_states_;
        return ComputeResult::RUNNING;
    }
}

void SAViveTracker::onStop(StopReason reason)
{
    model_updater_.haltCommands();
    front_overview_publish_enabled_.store(false, std::memory_order_relaxed);
    front_overview_publish_log_pending_.store(false, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(front_overview_publish_mutex_);
        last_front_overview_publish_time_ns_ = 0;
        front_overview_publish_until_ns_ = 0;
    }

    const char* reason_str = "none";
    if (reason == StopReason::CANCELED)
    {
        reason_str = "canceled";
    }
    else if (reason == StopReason::SUCCEEDED)
    {
        reason_str = "succeeded";
    }
    else if (reason == StopReason::ABORTED)
    {
        reason_str = "aborted";
    }

    RCLCPP_INFO(node_->get_logger(), "[%s] stopped (%s)", name_.c_str(), reason_str);
}

SAViveTracker::ResultPtr SAViveTracker::makeResult(StopReason reason)
{
    auto result = std::make_shared<ActionT::Result>();
    result->is_completed = true;
    return result;
}

void SAViveTracker::subPoseCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
{
    const double cutoff_freq = 100.;
    if(msg->poses.size() != NUM_TRACKERS)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Size of PoseArray for tracker_pose (%ld) does not equal to 3.", name_.c_str(), msg->poses.size());
    }
    else
    {
        for(size_t i = 0; i < msg->poses.size(); ++i)
        {
            Eigen::Vector3d position(msg->poses[i].position.x, msg->poses[i].position.y, msg->poses[i].position.z);
            position = dyros_math::lowPassFilter(position, controller_poses_[i].translation(), fr3_model_updater_.dt_, 1./cutoff_freq);
            Eigen::Quaterniond quaternion(msg->poses[i].orientation.w, msg->poses[i].orientation.x, msg->poses[i].orientation.y, msg->poses[i].orientation.z);
            quaternion.normalize();
            Eigen::Quaterniond prev_quat(controller_poses_[i].linear());
            const double alpha = fr3_model_updater_.dt_ / (fr3_model_updater_.dt_ + (1./cutoff_freq));
            Eigen::Quaterniond filtered_quat = prev_quat.slerp(alpha, quaternion);
            Eigen::Matrix3d orientation = filtered_quat.normalized().toRotationMatrix();
            {
                std::lock_guard<std::mutex> lock(tracker_pose_mutex_);
                controller_poses_[i].translation() = position;
                controller_poses_[i].linear() = orientation;
            }
        }
    }
}

void SAViveTracker::subLJoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    if(msg->buttons.size() != NUM_BUTTONS)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Size of buttons for lhand_joy (%ld) does not equal to %d.", name_.c_str(), msg->buttons.size(), NUM_BUTTONS);
    }
    else
    {
        for(size_t i = 0; i < msg->buttons.size(); ++i)
        {
            std::lock_guard<std::mutex> lock(button_state_mutex_);
            button_states_[IDX_LEFT_CON][i] = (static_cast<int>(msg->buttons[i]) == 0) ? false : true;
        }

    }
}

void SAViveTracker::subRJoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    if(msg->buttons.size() != NUM_BUTTONS)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Size of buttons for rhand_joy (%ld) does not equal to %d.", name_.c_str(), msg->buttons.size(), NUM_BUTTONS);
    }
    else
    {
        for(size_t i = 0; i < msg->buttons.size(); ++i)
        {
            std::lock_guard<std::mutex> lock(button_state_mutex_);
            button_states_[IDX_RIGHT_CON][i] = (static_cast<int>(msg->buttons[i]) == 0) ? false : true;
        }

    }
}

void SAViveTracker::subRightConstraintCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (msg->data.size() != 3)
    {
        RCLCPP_WARN(
            node_->get_logger(),
            "[%s] Size of Float64MultiArray for sa_right_eef_constraint (%ld) does not equal to 3.",
            name_.c_str(),
            msg->data.size());
        return;
    }

    Eigen::Vector3d constraint = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < 3; ++i)
    {
        constraint(static_cast<Eigen::Index>(i)) = msg->data[i];
    }

    std::lock_guard<std::mutex> lock(right_constraint_mutex_);
    const bool first_constraint = !right_constraint_received_;
    const bool changed = !right_constraint_received_ ||
                         !right_constraint_vector_.isApprox(constraint, 1e-9);
    right_constraint_vector_ = constraint;
    right_constraint_received_ = true;
    if (first_constraint)
    {
        right_constraint_applying_ = true;
        right_constraint_orientation_locked_ = false;
        right_constraint_anchor_pose_locked_ = false;
        RCLCPP_INFO(
            node_->get_logger(),
            "[%s] Received first right constraint -> applying immediately",
            name_.c_str());
    }
    else if (changed)
    {
        right_constraint_orientation_locked_ = false;
        if (right_constraint_applying_)
        {
            right_constraint_anchor_pose_locked_ = false;
        }
    }
}

void SAViveTracker::subFrontOverviewImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    if (!front_overview_publish_enabled_.load(std::memory_order_relaxed))
    {
        return;
    }

    auto resized = resizeImageNearest(*msg, 84, 84);
    if (resized.step == 0 || resized.data.empty())
    {
        return;
    }

    const int64_t now_ns = node_->now().nanoseconds();
    {
        std::lock_guard<std::mutex> lock(front_overview_publish_mutex_);
        if (front_overview_publish_until_ns_ != 0 && now_ns > front_overview_publish_until_ns_)
        {
            front_overview_publish_enabled_.store(false, std::memory_order_relaxed);
            return;
        }

        constexpr int64_t kPublishPeriodNs = 2000000000LL;
        if (last_front_overview_publish_time_ns_ != 0 &&
            (now_ns - last_front_overview_publish_time_ns_) < kPublishPeriodNs)
        {
            return;
        }

        last_front_overview_publish_time_ns_ = now_ns;
    }

    front_overview_image_pub_->publish(resized);

    if (front_overview_publish_log_pending_.exchange(false, std::memory_order_relaxed))
    {
        RCLCPP_INFO(node_->get_logger(),
                    "[%s] Started publishing resized front_overview images on sa_front_overview/image_raw",
                    name_.c_str());
    }
}


// Register this server into global registry (executed when this TU is linked)
REGISTER_FR3_ACTION_SERVER(SAViveTracker, "fr3_sa_vive_tracker")

}  // namespace fr3_husky_controller::servers::fr3
/*
# send goal 
ros2 action send_goal /fr3_sa_vive_tracker fr3_husky_msgs/action/ViveTracker \
"{mode: 1, left_controller_ee_name: 'left_fr3_hand_tcp', right_controller_ee_name: 'right_fr3_hand_tcp', move_orientation: false, controller_pos_multiplier: 1.0, controller_ori_multiplier: 1.0}" \
--feedback
*/
