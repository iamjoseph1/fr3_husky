#include <fr3_husky_controller/servers/fr3_husky/sa_apple_vision_pro_action_server.hpp>
#include <mujoco/mujoco.h>

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace mujoco_ros_hardware
{
class MujocoWorldSingleton
{
public:
    static MujocoWorldSingleton& get();
    bool isSceneLoaded() const;
    mjModel* model() const;
    mjData* data() const;
    std::mutex& dataMutex();
};
}  // namespace mujoco_ros_hardware

namespace fr3_husky_controller::servers::fr3_husky
{

namespace
{
FR3HuskyModelUpdater& getFR3HuskyModelUpdater(ModelUpdaterBase& model_updater, const std::string& server_name)
{
    auto* fr3_husky_model_updater = dynamic_cast<FR3HuskyModelUpdater*>(&model_updater);
    if (!fr3_husky_model_updater)
    {
        throw std::runtime_error("[" + server_name + "] requires FR3HuskyModelUpdater");
    }
    return *fr3_husky_model_updater;
}

// Extract robot name ("left" or "right") from an ee_name such as "left_fr3_hand_tcp".
// Returns an empty string if the prefix is not recognised.
std::string getRobotNameFromEEName(const std::string& ee_name)
{
    if (ee_name.rfind("left_", 0) == 0)  return "left";
    if (ee_name.rfind("right_", 0) == 0) return "right";
    return "";
}

Eigen::Matrix3d getBaseFromAVPPositionMap()
{
    Eigen::Matrix3d R;
    // Columns = AVP +x, +y, +z expressed in base_link
    R.col(0) = Eigen::Vector3d( 0.0, -1.0,  0.0);  // AVP +x -> base -y
    R.col(1) = Eigen::Vector3d( 0.0,  0.0,  1.0);  // AVP +y -> base +z
    R.col(2) = Eigen::Vector3d(-1.0,  0.0,  0.0);  // AVP +z -> base -x
    return R;
}

Eigen::Matrix3d getEEFfromHandRotationMap()
{
    Eigen::Matrix3d R;
    // Map hand local vectors into eef local vectors:
    // hand -x -> eef +z
    // hand +y -> eef +y
    // hand +z -> eef +x
    R <<
         0.0, 0.0, 1.0,
         0.0, 1.0, 0.0,
        -1.0, 0.0, 0.0;
    return R;
}

Eigen::Quaterniond averageQuaternionWXYZ(const Eigen::Vector4d& qsum)
{
    Eigen::Vector4d q = qsum;
    if (q.norm() < 1e-12)
    {
        return Eigen::Quaterniond::Identity();
    }
    q.normalize();
    return Eigen::Quaterniond(q(0), q(1), q(2), q(3)); // w, x, y, z
}

Eigen::Vector3d extractSelectedLocalAxis(const Eigen::Vector3d& selector)
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
    const Eigen::Vector3d local_axis = extractSelectedLocalAxis(local_rotation_selector);
    if (local_axis.norm() < 1e-6)
    {
        return locked_orientation;
    }

    const Eigen::Matrix3d relative_rotation = locked_orientation.transpose() * raw_orientation;
    Eigen::AngleAxisd relative_aa(relative_rotation);

    if (std::abs(relative_aa.angle()) < 1e-9 || relative_aa.axis().squaredNorm() < 1e-9)
    {
        return locked_orientation;
    }

    const Eigen::Vector3d relative_rotvec = relative_aa.axis() * relative_aa.angle();
    const Eigen::Vector3d projected_rotvec = projectVectorOntoAxis(relative_rotvec, local_axis);
    const double projected_angle = projected_rotvec.norm();
    if (projected_angle < 1e-9)
    {
        return locked_orientation;
    }

    const Eigen::Matrix3d constrained_relative =
        Eigen::AngleAxisd(projected_angle, projected_rotvec / projected_angle).toRotationMatrix();
    return locked_orientation * constrained_relative;
}

// ==================== MUJOCO OBJECT WELD ATTACH / DETACH ====================
// Predefined in fr3_husky_description/mjcf/dual_fr3_husky.xml.xacro:
//   weld_needle_right_tcp: right_fr3_hand_tcp <-> needle_obj
//   weld_needle_left_tcp:  left_fr3_hand_tcp  <-> needle_obj
// This directly toggles MuJoCo's equality constraint in the shared simulation.
// ============================================================================
bool setObjectTcpWeldActive(const rclcpp::Logger& logger,
                                       bool active,
                                       const std::vector<bool>& is_gripper_mode_on,
                                       int controller_idx)
{
    auto& world = mujoco_ros_hardware::MujocoWorldSingleton::get();
    if (!world.isSceneLoaded())
    {
        RCLCPP_WARN(logger, "[AVP object weld] MuJoCo scene is not loaded; cannot %s weld.",
                    active ? "attach" : "detach");
        return false;
    }

    std::lock_guard<std::mutex> lock(world.dataMutex());
    mjModel* model = world.model();
    mjData* data = world.data();
    if (!model || !data)
    {
        RCLCPP_WARN(logger, "[AVP object weld] MuJoCo model/data unavailable.");
        return false;
    }

    const bool is_right_controller = controller_idx == IDX_RIGHT_CON;
    const bool is_left_controller = controller_idx == IDX_LEFT_CON;
    if (!is_right_controller && !is_left_controller)
    {
        RCLCPP_WARN(logger, "[AVP object weld] Invalid controller index: %d", controller_idx);
        return false;
    }

    if (active && (controller_idx >= static_cast<int>(is_gripper_mode_on.size()) ||
                   !is_gripper_mode_on[controller_idx]))
    {
        RCLCPP_WARN(logger,
                    "[AVP object weld] Cannot attach; gripper mode is off for controller index %d.",
                    controller_idx);
        return false;
    }

    const char* const kWeldName = is_right_controller ? "weld_needle_right_tcp" : "weld_needle_left_tcp";
    const char* const kParentBodyName = is_right_controller ? "right_fr3_hand_tcp" : "left_fr3_hand_tcp";
    constexpr const char* kChildBodyName = "needle_obj";
    constexpr const char* kChildFreeJointName = "needle_obj_joint";

    const int weld_id = mj_name2id(model, mjOBJ_EQUALITY, kWeldName);
    const int parent_body_id = mj_name2id(model, mjOBJ_BODY, kParentBodyName);
    const int child_body_id = mj_name2id(model, mjOBJ_BODY, kChildBodyName);

    if (weld_id < 0 || parent_body_id < 0 || child_body_id < 0)
    {
        RCLCPP_WARN(logger,
                    "[AVP object weld] Missing weld/body. weld=%d parent(%s)=%d child(%s)=%d",
                    weld_id, kParentBodyName, parent_body_id, kChildBodyName, child_body_id);
        return false;
    }

    if (active)
    {
        // Update weld relative pose at the instant of attachment. This avoids a
        // large snap impulse from using the XML compile-time relative pose.
        mjtNum rel_pos_world[3];
        mjtNum rel_pos_parent[3];
        mju_sub3(rel_pos_world, data->xpos + 3 * child_body_id, data->xpos + 3 * parent_body_id);
        mju_mulMatTVec(rel_pos_parent, data->xmat + 9 * parent_body_id, rel_pos_world, 3, 3);

        mjtNum parent_quat_inv[4];
        mjtNum rel_quat[4];
        mju_negQuat(parent_quat_inv, data->xquat + 4 * parent_body_id);
        mju_mulQuat(rel_quat, parent_quat_inv, data->xquat + 4 * child_body_id);
        const mjtNum rel_quat_norm = std::sqrt(rel_quat[0] * rel_quat[0] +
                                               rel_quat[1] * rel_quat[1] +
                                               rel_quat[2] * rel_quat[2] +
                                               rel_quat[3] * rel_quat[3]);
        if (rel_quat_norm > mjtNum(1e-12))
        {
            for (int i = 0; i < 4; ++i)
            {
                rel_quat[i] /= rel_quat_norm;
            }
        }

        mjtNum* eq_data = model->eq_data + weld_id * mjNEQDATA;
        // MuJoCo weld eq_data layout is [anchor(3), relpos(3), relquat(4), ...].
        // Match TMM's attachment logic: update only the current relative pose,
        // then enable the predefined weld. Do not move either body at attach time.
        eq_data[3] = rel_pos_parent[0];
        eq_data[4] = rel_pos_parent[1];
        eq_data[5] = rel_pos_parent[2];
        eq_data[6] = rel_quat[0];
        eq_data[7] = rel_quat[1];
        eq_data[8] = rel_quat[2];
        eq_data[9] = rel_quat[3];

        const int free_joint_id = mj_name2id(model, mjOBJ_JOINT, kChildFreeJointName);
        if (free_joint_id >= 0)
        {
            const int dof_adr = model->jnt_dofadr[free_joint_id];
            for (int i = 0; i < 6; ++i)
            {
                data->qvel[dof_adr + i] = 0.0;
            }
        }
    }

    data->eq_active[weld_id] = active ? 1 : 0;
    mj_forward(model, data);
    RCLCPP_INFO(logger, "[AVP object weld] %s %s: %s <-> %s",
                active ? "Attached" : "Detached", kWeldName, kParentBodyName, kChildBodyName);
    return true;
}

}  // namespace

SAAppleVisionPro::SAAppleVisionPro(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater)
: Base(name, node, model_updater),
  fr3_husky_model_updater_(getFR3HuskyModelUpdater(model_updater, name))
{
    const auto tracker_pose_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    pose_sub_ = node_->create_subscription<geometry_msgs::msg::PoseArray>(
        "tracker_pose",
        tracker_pose_qos,
        std::bind(&SAAppleVisionPro::subPoseCallback, this, std::placeholders::_1));
    right_constraint_sub_ = node_->create_subscription<std_msgs::msg::Float64MultiArray>(
        "sa_right_eef_constraint",
        rclcpp::QoS(1).best_effort(),
        std::bind(&SAAppleVisionPro::subRightConstraintCallback, this, std::placeholders::_1));
    l_gesture_state_sub_ = node_->create_subscription<std_msgs::msg::Int32MultiArray>("lhand_gesture", 1, std::bind(&SAAppleVisionPro::subLGestureCallback, this, std::placeholders::_1));
    r_gesture_state_sub_ = node_->create_subscription<std_msgs::msg::Int32MultiArray>("rhand_gesture", 1, std::bind(&SAAppleVisionPro::subRGestureCallback, this, std::placeholders::_1));

    controller_poses_.assign(NUM_TRACKERS, Eigen::Affine3d::Identity());
    controller_poses_init_.assign(NUM_TRACKERS, Eigen::Affine3d::Identity());
    gesture_states_.assign(NUM_CONTROLLERS, std::vector<bool>(NUM_GESTURES, false));
    prev_gesture_states_.assign(NUM_CONTROLLERS, std::vector<bool>(NUM_GESTURES, false));

    tracker_base2robot_base_.assign(NUM_CONTROLLERS, Eigen::Matrix3d::Identity());

    ee_data_.clear();

    // Action clients
    move_to_joint_client_ = rclcpp_action::create_client<MoveToJointAction>(node_, "fr3_move_to_joint");
    avp_self_client_       = rclcpp_action::create_client<ActionT>(node_, name_);

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
                            "[%s] JTC finished — re-activating SAAppleVisionPro", name_.c_str());
                avp_self_client_->async_send_goal(saved_avp_goal_);
            }
        });

    // Initialize franka hand state
    for(const auto& robot_name : fr3_husky_model_updater_.robot_names_) fr3_husky_model_updater_.GripperHoming(robot_name); 

    RCLCPP_INFO(node_->get_logger(), "[%s] SAAppleVisionPro created", name_.c_str());
}

bool SAAppleVisionPro::acceptGoal(const ActionT::Goal& goal)
{


    if (!fr3_husky_model_updater_.HasEffortCommandInterface())
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

    if(!goal.left_controller_ee_name.empty() && !fr3_husky_model_updater_.robot_data_->hasLinkFrame(goal.left_controller_ee_name))
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: left_controller_ee_name from the goal [%s] is not includede in URDF.",
                                         name_.c_str(), goal.left_controller_ee_name.c_str());
        return false;
    }

    if(!goal.right_controller_ee_name.empty() && !fr3_husky_model_updater_.robot_data_->hasLinkFrame(goal.right_controller_ee_name))
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: right_controller_ee_name from the goal [%s] is not includede in URDF.",
                                         name_.c_str(), goal.right_controller_ee_name.c_str());
        return false;
    }

    return true;
}

void SAAppleVisionPro::onGoalAccepted(const ActionT::Goal& goal)
{
    control_mode_ = goal.mode;
    left_controller_ee_name_ = goal.left_controller_ee_name;
    right_controller_ee_name_ = goal.right_controller_ee_name;
    move_ori_ = goal.move_orientation;
    controller_pos_multiplier_ = static_cast<double>(goal.controller_pos_multiplier);
    controller_ori_multiplier_ = static_cast<double>(goal.controller_ori_multiplier);
    saved_avp_goal_ = goal;

    requestActivate();
}


void SAAppleVisionPro::onStart()
{
    {
        std::lock_guard<std::mutex> lock(tracker_pose_mutex_);
        for(auto& tracker_pose : controller_poses_) tracker_pose.setIdentity();
    }
    for(auto& tracker_pose_init : controller_poses_init_) tracker_pose_init.setIdentity();
    {
        std::lock_guard<std::mutex> lock(gesture_state_mutex_);
        for(auto& gesture_state : gesture_states_) gesture_state = std::vector<bool>(NUM_GESTURES, false);
    }

    for(auto& prev_gesture_state : prev_gesture_states_) prev_gesture_state = std::vector<bool>(NUM_GESTURES, false);
    is_tracking_mode_on_.assign(NUM_CONTROLLERS, false);
    is_initialize_mode_on_ = false;
    is_gripper_mode_on_.assign(NUM_CONTROLLERS, false);

    // tracking state
    auto_tracking_started_ = false;
    tracker_pose_valid_.assign(NUM_TRACKERS, false);

    ee_data_.clear();
    waiting_for_jtc_.store(false, std::memory_order_relaxed);
    control_start_time_ = -1.0; // sentinel: set on first compute() call
    q_init_for_home_ = fr3_husky_model_updater_.q_total_;
    right_constraint_orientation_locked_ = false;

    if(!left_controller_ee_name_.empty())
    {
        ee_data_[left_controller_ee_name_] = drc::TaskSpaceData::Zero();
        ee_data_[left_controller_ee_name_].x = fr3_husky_model_updater_.robot_data_->getPose(left_controller_ee_name_);
        ee_data_[left_controller_ee_name_].xdot = fr3_husky_model_updater_.robot_data_->getVelocity(left_controller_ee_name_);
        ee_data_[left_controller_ee_name_].xddot.setZero();
        ee_data_[left_controller_ee_name_].setInit();
        ee_data_[left_controller_ee_name_].setDesired();
    }
    if(!right_controller_ee_name_.empty())
    {
        ee_data_[right_controller_ee_name_] = drc::TaskSpaceData::Zero();
        ee_data_[right_controller_ee_name_].x = fr3_husky_model_updater_.robot_data_->getPose(right_controller_ee_name_);
        ee_data_[right_controller_ee_name_].xdot = fr3_husky_model_updater_.robot_data_->getVelocity(right_controller_ee_name_);
        ee_data_[right_controller_ee_name_].xddot.setZero();
        ee_data_[right_controller_ee_name_].setInit();
        ee_data_[right_controller_ee_name_].setDesired();
    }

    RCLCPP_INFO(node_->get_logger(), "[%s] started", name_.c_str());
}

SAAppleVisionPro::ComputeResult SAAppleVisionPro::compute(const rclcpp::Time& time, const rclcpp::Duration& /*period*/)
{


    for(auto& [ee_name, ee_data] : ee_data_)
    {
        ee_data.x = fr3_husky_model_updater_.robot_data_->getPose(ee_name);
        ee_data.xdot = fr3_husky_model_updater_.robot_data_->getVelocity(ee_name);
        ee_data.xddot.setZero();
    }

    std::vector<Eigen::Affine3d> controller_poses_local;   // left, right, head
    std::vector<std::vector<bool>> gesture_states_local;    // [left, right]
    {
        std::lock_guard<std::mutex> lock(tracker_pose_mutex_);
        controller_poses_local = controller_poses_;
    }
    {
        std::lock_guard<std::mutex> lock(gesture_state_mutex_);
        gesture_states_local = gesture_states_;
    }
    Eigen::Matrix<double, 6, 1> right_constraint_vector_local = Eigen::Matrix<double, 6, 1>::Zero();
    bool right_constraint_received_local = false;
    {
        std::lock_guard<std::mutex> lock(right_constraint_mutex_);
        right_constraint_vector_local = right_constraint_vector_;
        right_constraint_received_local = right_constraint_received_;
    }


    if (control_start_time_ < 0) 
    {
        control_start_time_ = time.seconds();
        q_init_for_home_ = fr3_husky_model_updater_.q_total_;
    }

    // Initialize mode
    {
        if (!is_initialize_mode_on_)
        {
            // if "a" button on the AVP controller is pressed
            if ((!prev_gesture_states_[IDX_LEFT_CON][IDX_PINCH_SNAP_LEFT_GESTURE]  && gesture_states_local[IDX_LEFT_CON][IDX_PINCH_SNAP_LEFT_GESTURE]) ||
                (!prev_gesture_states_[IDX_RIGHT_CON][IDX_PINCH_SNAP_LEFT_GESTURE] && gesture_states_local[IDX_RIGHT_CON][IDX_PINCH_SNAP_LEFT_GESTURE]))
            {
                is_initialize_mode_on_ = true;

                MoveToJointAction::Goal mtj_goal;
                for (const auto& robot_name : fr3_husky_model_updater_.robot_names_)
                {
                    for (size_t j = 0; j < FR3_DOF; ++j)
                    {
                        mtj_goal.joint_names.push_back(robot_name + "_" + fr3_husky_model_updater_.arm_id_ + "_joint" + std::to_string(j+1));
                        mtj_goal.target_positions.push_back(HomePose(j));
                    }
                }
                mtj_goal.max_velocity_scaling_factor     = 0.1;
                mtj_goal.max_acceleration_scaling_factor = 0.1;

                // When MoveToJoint succeeds (trajectory sent to JTC), set waiting_for_jtc_ so
                // the JTC status subscriber re-activates SAAppleVisionPro after the robot finishes moving.
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

                // Yield: deactivate SAAppleVisionPro so MoveToJoint can become active_server_
                return ComputeResult::SUCCEEDED;
            }
        }
    }

    // Gripper control
    {
        // Left controller
        if (!left_controller_ee_name_.empty())
        {
            if (!prev_gesture_states_[IDX_LEFT_CON][IDX_DOUBLE_TAP_GESTURE] && gesture_states_local[IDX_LEFT_CON][IDX_DOUBLE_TAP_GESTURE])
            {
                const std::string robot_name = getRobotNameFromEEName(left_controller_ee_name_);
                if (!robot_name.empty())
                {
                    is_gripper_mode_on_[IDX_LEFT_CON] = !is_gripper_mode_on_[IDX_LEFT_CON];
                    if (is_gripper_mode_on_[IDX_LEFT_CON])
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] lhand trigger released → GripperGrasp('%s')", name_.c_str(), robot_name.c_str());
                        fr3_husky_model_updater_.GripperGrasp(robot_name, 0.0, 0.1, 100.0);

                        // Attach object to left_fr3_hand_tcp while the left gripper is closed.
                        setObjectTcpWeldActive(node_->get_logger(), false, is_gripper_mode_on_, IDX_RIGHT_CON);
                        setObjectTcpWeldActive(node_->get_logger(), true, is_gripper_mode_on_, IDX_LEFT_CON);
                    }
                    else
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] lhand trigger released → GripperOpen('%s')", name_.c_str(), robot_name.c_str());

                        // Detach object from left_fr3_hand_tcp before opening the left gripper.
                        setObjectTcpWeldActive(node_->get_logger(), false, is_gripper_mode_on_, IDX_LEFT_CON);

                        fr3_husky_model_updater_.GripperOpen(robot_name, 0.1);
                    }
                }
            }
        }

        // Right controller
        if (!right_controller_ee_name_.empty())
        {
            if (!prev_gesture_states_[IDX_RIGHT_CON][IDX_DOUBLE_TAP_GESTURE] && gesture_states_local[IDX_RIGHT_CON][IDX_DOUBLE_TAP_GESTURE])
            {
                const std::string robot_name = getRobotNameFromEEName(right_controller_ee_name_);
                if (!robot_name.empty())
                {
                    is_gripper_mode_on_[IDX_RIGHT_CON] = !is_gripper_mode_on_[IDX_RIGHT_CON];
                    if (is_gripper_mode_on_[IDX_RIGHT_CON])
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] rhand trigger released → GripperGrasp('%s')", name_.c_str(), robot_name.c_str());
                        fr3_husky_model_updater_.GripperGrasp(robot_name, 0.0, 0.1, 100.0);

                        // Attach object to right_fr3_hand_tcp while the right gripper is closed.
                        setObjectTcpWeldActive(node_->get_logger(), false, is_gripper_mode_on_, IDX_LEFT_CON);
                        setObjectTcpWeldActive(node_->get_logger(), true, is_gripper_mode_on_, IDX_RIGHT_CON);
                    }
                    else
                    {
                        RCLCPP_INFO(node_->get_logger(), "[%s] rhand trigger released → GripperOpen('%s')", name_.c_str(), robot_name.c_str());

                        // Detach object from right_fr3_hand_tcp before opening the right gripper.
                        setObjectTcpWeldActive(node_->get_logger(), false, is_gripper_mode_on_, IDX_RIGHT_CON);

                        fr3_husky_model_updater_.GripperOpen(robot_name, 0.1);
                    }
                }
            }
        }
    }

    // Manipulator control
    {   

        if (!auto_tracking_started_)
        {
            const bool head_tracker_valid =
                tracker_pose_valid_.size() == NUM_TRACKERS &&
                tracker_pose_valid_[IDX_HEAD_CON];

            const bool left_tracker_valid =
                head_tracker_valid && tracker_pose_valid_[IDX_LEFT_CON];

            const bool right_tracker_valid =
                head_tracker_valid && tracker_pose_valid_[IDX_RIGHT_CON];

            if ((left_tracker_valid || right_tracker_valid) &&
                control_start_time_ >= 0.0 &&
                (time.seconds() - control_start_time_) >= avp_tracking_enable_delay_)
            {
                RCLCPP_INFO(node_->get_logger(),
                            "[%s] Auto tracking ON after %.2f sec. left=%s right=%s",
                            name_.c_str(),
                            avp_tracking_enable_delay_,
                            left_tracker_valid ? "true" : "false",
                            right_tracker_valid ? "true" : "false");

                // Head init is common
                controller_poses_init_[IDX_HEAD_CON] = controller_poses_local[IDX_HEAD_CON];

                // Left hand / left EEF
                is_tracking_mode_on_[IDX_LEFT_CON] = left_tracker_valid;
                if (left_tracker_valid && !left_controller_ee_name_.empty())
                {
                    controller_poses_init_[IDX_LEFT_CON] = controller_poses_local[IDX_LEFT_CON];
                    ee_data_[left_controller_ee_name_].setInit();
                    is_first_target_left_ = true;
                }

                // Right hand / right EEF
                is_tracking_mode_on_[IDX_RIGHT_CON] = right_tracker_valid;
                if (right_tracker_valid && !right_controller_ee_name_.empty())
                {
                    controller_poses_init_[IDX_RIGHT_CON] = controller_poses_local[IDX_RIGHT_CON];
                    ee_data_[right_controller_ee_name_].setInit();
                    is_first_target_right_ = true;
                }

                // world_from_base_init_ = fr3_husky_model_updater_.robot_data_->getPose("base_link");
                auto_tracking_started_ = true;
            }
        }
        
        
        if(!left_controller_ee_name_.empty()) // left AVP controller
        {
            Eigen::Affine3d target_pose_diff; // EE init -> EE desired
            Eigen::Vector6d target_vel;
            target_pose_diff.setIdentity();
            target_vel.setZero();

            if (is_tracking_mode_on_[IDX_LEFT_CON])
            {
                // if (!runStartupOrientationCalibration(IDX_LEFT_CON, left_controller_ee_name_))
                if (false)
                {
                    target_pose_diff.setIdentity();
                }
                else
                {
                    const Eigen::Matrix3d R_base_from_avp = getBaseFromAVPPositionMap();

                    // ---------------------------
                    // POSITION CALIBRATION (TRUE HEAD-RELATIVE)
                    // ---------------------------
                    const Eigen::Vector3d p_hand_cur_world =
                        controller_poses_local[IDX_LEFT_CON].translation();
                    const Eigen::Vector3d p_hand_init_world =
                        controller_poses_init_[IDX_LEFT_CON].translation();

                    const Eigen::Vector3d p_head_cur_world =
                        controller_poses_local[IDX_HEAD_CON].translation();
                    const Eigen::Vector3d p_head_init_world =
                        controller_poses_init_[IDX_HEAD_CON].translation();

                    const Eigen::Matrix3d R_world_from_head_cur =
                        controller_poses_local[IDX_HEAD_CON].linear();
                    const Eigen::Matrix3d R_world_from_head_init =
                        controller_poses_init_[IDX_HEAD_CON].linear();

                    // Hand expressed in current / initial head frame
                    const Eigen::Vector3d hand_cur_rel_head =
                        R_world_from_head_cur.transpose() * (p_hand_cur_world - p_head_cur_world);

                    const Eigen::Vector3d hand_init_rel_head =
                        R_world_from_head_init.transpose() * (p_hand_init_world - p_head_init_world);

                    // True AVP-frame delta
                    Eigen::Vector3d delta_avp =
                        hand_cur_rel_head - hand_init_rel_head;

                    // Deadband
                    const double POS_EPS = 0.05;
                    for (int k = 0; k < 3; ++k)
                    {
                        if (std::abs(delta_avp(k)) < POS_EPS)
                            delta_avp(k) = 0.0;
                    }

                    // Clamp
                    const double MAX_POS_DELTA = 0.5;
                    for (int k = 0; k < 3; ++k)
                    {
                        if (delta_avp(k) >  MAX_POS_DELTA) delta_avp(k) =  MAX_POS_DELTA;
                        if (delta_avp(k) < -MAX_POS_DELTA) delta_avp(k) = -MAX_POS_DELTA;
                    }

                    // Map AVP frame -> base frame
                    const Eigen::Vector3d delta_base =
                        R_base_from_avp * delta_avp;

                    // Get current base link pose in world frame
                    world_from_base_cur_ = fr3_husky_model_updater_.robot_data_->getPose("base_link");

                    // Convert base delta to world using base pose at tracking start
                    const Eigen::Vector3d delta_world =
                        world_from_base_cur_.linear() * delta_base;

                    target_pose_diff.translation() =
                        controller_pos_multiplier_ *
                        ee_data_[left_controller_ee_name_].x_init.linear().transpose() *
                        delta_world;

                    // ---------------------------
                    // ORIENTATION CALIBRATION
                    // Hand AVP-axis delta -> EEF WORLD(=base_init)-axis delta
                    // ---------------------------
                    target_pose_diff.linear().setIdentity();
                    if (move_ori_)
                    {
                        // Hand orientations are expressed in AVP frame, NOT world frame
                        const Eigen::Matrix3d R_hand_init_avp =
                            controller_poses_init_[IDX_LEFT_CON].linear();
                        const Eigen::Matrix3d R_hand_cur_avp =
                            controller_poses_local[IDX_LEFT_CON].linear();

                        // Spatial delta in AVP frame
                        const Eigen::Matrix3d R_hand_delta_avp =
                            R_hand_cur_avp * R_hand_init_avp.transpose();

                        Eigen::AngleAxisd aa_hand(R_hand_delta_avp);
                        Eigen::Matrix3d R_hand_delta_avp_scaled = Eigen::Matrix3d::Identity();

                        double angle = aa_hand.angle();

                        const double ROT_EPS = 0.05;   // ~3 deg
                        if (std::abs(angle) < ROT_EPS)
                        {
                            angle = 0.0;
                        }

                        const double MAX_ROT_DELTA = 0.60;  // ~34 deg
                        if (angle >  MAX_ROT_DELTA) angle =  MAX_ROT_DELTA;
                        if (angle < -MAX_ROT_DELTA) angle = -MAX_ROT_DELTA;

                        if (std::abs(angle) > 1e-10)
                        {
                            R_hand_delta_avp_scaled =
                                Eigen::AngleAxisd(
                                    controller_ori_multiplier_ * angle,
                                    aa_hand.axis()
                                ).toRotationMatrix();
                        }

                        // Convert AVP-axis delta to base_init-axis delta
                        const Eigen::Matrix3d R_base_from_eef_delta =
                            R_base_from_avp *
                            R_hand_delta_avp_scaled *
                            R_base_from_avp.transpose();

                        // Convert base_init-axis delta to world-axis delta
                        const Eigen::Matrix3d R_world_from_eef_delta =
                            world_from_base_cur_.linear() *
                            R_base_from_eef_delta *
                            world_from_base_cur_.linear().transpose();
                        
                        // Apply same world/base-axis delta to initial EEF orientation
                        const Eigen::Matrix3d R_world_from_eef_des =
                            R_world_from_eef_delta * ee_data_[left_controller_ee_name_].x_init.linear();

                        // raw_target = x_init * target_pose_diff
                        target_pose_diff.linear() =
                            ee_data_[left_controller_ee_name_].x_init.linear().transpose() *
                            R_world_from_eef_des;
                    }
                
                // smoothed target pose
                Eigen::Affine3d raw_target = ee_data_[left_controller_ee_name_].x_init * target_pose_diff;
                double dt = fr3_husky_model_updater_.dt_;
                if (is_first_target_left_)
                {
                    prev_target_left_ = raw_target;
                    is_first_target_left_ = false;
                }
                Eigen::Affine3d smooth_target = smoothAndLimit(prev_target_left_, raw_target, dt);
                prev_target_left_ = smooth_target;
                ee_data_[left_controller_ee_name_].x_desired = smooth_target;
                ee_data_[left_controller_ee_name_].xdot_desired  = target_vel;
                
                }
            }
        }
    
        if(!right_controller_ee_name_.empty()) // right AVP controller
        {
            Eigen::Affine3d target_pose_diff; // EE init -> EE desired
            Eigen::Vector6d target_vel;
            target_pose_diff.setIdentity();
            target_vel.setZero();

            if (is_tracking_mode_on_[IDX_RIGHT_CON])
            {
                if (false)
                {
                    target_pose_diff.setIdentity();
                }
                else
                {
                    const Eigen::Matrix3d R_base_from_avp = getBaseFromAVPPositionMap();

                    // ---------------------------
                    // POSITION CALIBRATION (TRUE HEAD-RELATIVE)
                    // ---------------------------
                    const Eigen::Vector3d p_hand_cur_world =
                        controller_poses_local[IDX_RIGHT_CON].translation();
                    const Eigen::Vector3d p_hand_init_world =
                        controller_poses_init_[IDX_RIGHT_CON].translation();

                    const Eigen::Vector3d p_head_cur_world =
                        controller_poses_local[IDX_HEAD_CON].translation();
                    const Eigen::Vector3d p_head_init_world =
                        controller_poses_init_[IDX_HEAD_CON].translation();

                    const Eigen::Matrix3d R_world_from_head_cur =
                        controller_poses_local[IDX_HEAD_CON].linear();
                    const Eigen::Matrix3d R_world_from_head_init =
                        controller_poses_init_[IDX_HEAD_CON].linear();

                    const Eigen::Vector3d hand_cur_rel_head =
                        R_world_from_head_cur.transpose() * (p_hand_cur_world - p_head_cur_world);

                    const Eigen::Vector3d hand_init_rel_head =
                        R_world_from_head_init.transpose() * (p_hand_init_world - p_head_init_world);

                    Eigen::Vector3d delta_avp =
                        hand_cur_rel_head - hand_init_rel_head;

                    // Deadband
                    const double POS_EPS = 0.05;
                    for (int k = 0; k < 3; ++k)
                    {
                        if (std::abs(delta_avp(k)) < POS_EPS)
                            delta_avp(k) = 0.0;
                    }

                    // Clamp
                    const double MAX_POS_DELTA = 0.5;
                    for (int k = 0; k < 3; ++k)
                    {
                        if (delta_avp(k) >  MAX_POS_DELTA) delta_avp(k) =  MAX_POS_DELTA;
                        if (delta_avp(k) < -MAX_POS_DELTA) delta_avp(k) = -MAX_POS_DELTA;
                    }

                    const Eigen::Vector3d delta_base =
                        R_base_from_avp * delta_avp;
                    
                    // Get current base link pose in world frame
                    world_from_base_cur_ = fr3_husky_model_updater_.robot_data_->getPose("base_link");

                    const Eigen::Vector3d delta_world =
                        world_from_base_cur_.linear() * delta_base;

                    target_pose_diff.translation() =
                        controller_pos_multiplier_ *
                        ee_data_[right_controller_ee_name_].x_init.linear().transpose() *
                        delta_world;
                        
                    // ---------------------------
                    // ORIENTATION CALIBRATION
                    // Hand AVP-axis delta -> EEF WORLD(=base_init)-axis delta
                    // ---------------------------
                    target_pose_diff.linear().setIdentity();
                    if (move_ori_)
                    {
                        // Hand orientations are expressed in AVP frame, NOT world frame
                        const Eigen::Matrix3d R_hand_init_avp =
                            controller_poses_init_[IDX_RIGHT_CON].linear();
                        const Eigen::Matrix3d R_hand_cur_avp =
                            controller_poses_local[IDX_RIGHT_CON].linear();

                        // Spatial delta in AVP frame
                        const Eigen::Matrix3d R_hand_delta_avp =
                            R_hand_cur_avp * R_hand_init_avp.transpose();

                        Eigen::AngleAxisd aa_hand(R_hand_delta_avp);
                        Eigen::Matrix3d R_hand_delta_avp_scaled = Eigen::Matrix3d::Identity();

                        double angle = aa_hand.angle();

                        const double ROT_EPS = 0.05;   // ~3 deg
                        if (std::abs(angle) < ROT_EPS)
                        {
                            angle = 0.0;
                        }

                        const double MAX_ROT_DELTA = 0.60;  // ~34 deg
                        if (angle >  MAX_ROT_DELTA) angle =  MAX_ROT_DELTA;
                        if (angle < -MAX_ROT_DELTA) angle = -MAX_ROT_DELTA;

                        if (std::abs(angle) > 1e-10)
                        {
                            R_hand_delta_avp_scaled =
                                Eigen::AngleAxisd(
                                    controller_ori_multiplier_ * angle,
                                    aa_hand.axis()
                                ).toRotationMatrix();
                        }


                        // Convert AVP-axis delta to base_init-axis delta
                        const Eigen::Matrix3d R_base_from_eef_delta =
                            R_base_from_avp *
                            R_hand_delta_avp_scaled *
                            R_base_from_avp.transpose();

                        // Convert base_init-axis delta to world-axis delta
                        const Eigen::Matrix3d R_world_from_eef_delta =
                            world_from_base_cur_.linear() *
                            R_base_from_eef_delta *
                            world_from_base_cur_.linear().transpose();
                        
                        // Apply same world/base-axis delta to initial EEF orientation
                        const Eigen::Matrix3d R_world_from_eef_des =
                            R_world_from_eef_delta * ee_data_[right_controller_ee_name_].x_init.linear();

                        // raw_target = x_init * target_pose_diff
                        target_pose_diff.linear() =
                            ee_data_[right_controller_ee_name_].x_init.linear().transpose() *
                            R_world_from_eef_des;
                    }
                }                

                // smoothed target pose
                Eigen::Affine3d raw_target = ee_data_[right_controller_ee_name_].x_init * target_pose_diff;
                double dt = fr3_husky_model_updater_.dt_;
                if (is_first_target_right_)
                {
                    prev_target_right_ = raw_target;
                    is_first_target_right_ = false;
                }
                if (right_constraint_received_local)
                {
                    Eigen::Matrix3d locked_orientation = Eigen::Matrix3d::Identity();
                    {
                        std::lock_guard<std::mutex> lock(right_constraint_mutex_);
                        if (!right_constraint_orientation_locked_)
                        {
                            right_constraint_locked_orientation_ = ee_data_[right_controller_ee_name_].x.linear();
                            right_constraint_orientation_locked_ = true;
                        }
                        locked_orientation = right_constraint_locked_orientation_;
                    }

                    const Eigen::Vector3d translation_selector =
                        right_constraint_vector_local.head<3>();
                    const Eigen::Vector3d rotation_selector =
                        right_constraint_vector_local.tail<3>();

                    const Eigen::Vector3d local_translation_axis =
                        extractSelectedLocalAxis(translation_selector);
                    if (local_translation_axis.norm() > 1e-6)
                    {
                        const Eigen::Matrix3d R_world_from_base = world_from_base_cur_.linear();
                        const Eigen::Vector3d delta_world_raw =
                            raw_target.translation() - ee_data_[right_controller_ee_name_].x_init.translation();
                        const Eigen::Vector3d delta_base_raw =
                            R_world_from_base.transpose() * delta_world_raw;
                        const Eigen::Vector3d allowed_axis_base =
                            R_world_from_base.transpose() * (locked_orientation * local_translation_axis);
                        const Eigen::Vector3d delta_base_regulated =
                            projectVectorOntoAxis(delta_base_raw, allowed_axis_base);

                        raw_target.translation() =
                            ee_data_[right_controller_ee_name_].x_init.translation() +
                            R_world_from_base * delta_base_regulated;
                    }

                    raw_target.linear() = buildConstrainedOrientation(
                        locked_orientation,
                        raw_target.linear(),
                        rotation_selector);
                }

                Eigen::Affine3d smooth_target = smoothAndLimit(prev_target_right_, raw_target, dt);
                prev_target_right_ = smooth_target;
                ee_data_[right_controller_ee_name_].x_desired = smooth_target;
                ee_data_[right_controller_ee_name_].xdot_desired  = target_vel;
                }
            }
        }
    
        bool is_qp_solved = true;
        std::string time_verbose = "";

        Eigen::VectorXd qdot_mobile = Eigen::VectorXd::Zero(fr3_husky_model_updater_.mobile_dof_);

        switch (control_mode_)
        {
            case 0: // CLIK
                {   
                    // --- Arm null_qdot: cubic toward HomePose ---
                    Eigen::VectorXd HomePose_total(fr3_husky_model_updater_.manipulator_dof_);
                    for(size_t i = 0; i < fr3_husky_model_updater_.num_robots_; ++i)
                        HomePose_total.segment(FR3_DOF*i, FR3_DOF) = HomePose;

                    static constexpr double null_space_duration = 5.0;
                    const Eigen::VectorXd zeros_mani = Eigen::VectorXd::Zero(fr3_husky_model_updater_.manipulator_dof_);
                    const Eigen::VectorXd null_qdot_mani =
                        fr3_husky_model_updater_.robot_controller_->moveManipulatorJointVelocityCubic(HomePose_total, zeros_mani, q_init_for_home_,  zeros_mani, time.seconds(), control_start_time_, null_space_duration);

                    // --- Mobile null_qdot: drive mobile toward EE target (compensates for arm homing) ---
                    // As arm is pulled toward home by null space, EE error grows → mobile drives to fill the gap
                    Eigen::Vector3d avg_ee_pos_error = Eigen::Vector3d::Zero();
                    int ee_count = 0;
                    for(const auto& [ee_name, ee_data] : ee_data_)
                    {
                        avg_ee_pos_error += ee_data.x_desired.translation() - ee_data.x.translation();
                        ++ee_count;
                    }
                    if(ee_count > 0) avg_ee_pos_error /= ee_count;


                    const Eigen::Affine3d world2base_cur = fr3_husky_model_updater_.robot_data_->getPose("base_link");
                    const Eigen::Vector3d ee_error_base = world2base_cur.linear().transpose() * avg_ee_pos_error;
                    static constexpr double mobile_null_gain = 10.0; // [wheel_vel/m]: tune as needed
                    Eigen::Vector3d base_vel_null;
                    base_vel_null << ee_error_base(0), 0.0, 0.0; // Husky cannot strafe (y=0)
                    // const Eigen::VectorXd null_qdot_mobile =
                    //     fr3_husky_model_updater_.robot_controller_->MobileVelocityCommand(mobile_null_gain * base_vel_null);

                    const Eigen::VectorXd null_qdot_mobile =
                        Eigen::VectorXd::Zero(fr3_husky_model_updater_.mobile_dof_);

                    // --- Assemble null_qdot via ActuatorIndex ---
                    Eigen::VectorXd null_qdot(fr3_husky_model_updater_.robot_data_->getActuatorDof());
                    const auto& act_idx = fr3_husky_model_updater_.robot_data_->getActuatorIndex();
                    null_qdot.segment(act_idx.mobi_start, fr3_husky_model_updater_.mobile_dof_) = null_qdot_mobile;
                    null_qdot.segment(act_idx.mani_start, fr3_husky_model_updater_.manipulator_dof_)      = null_qdot_mani;

                    fr3_husky_model_updater_.robot_controller_->CLIKStep(ee_data_, qdot_mobile, fr3_husky_model_updater_.qdot_desired_total_, null_qdot);
                    fr3_husky_model_updater_.q_desired_total_ = fr3_husky_model_updater_.q_total_ +
                                                                fr3_husky_model_updater_.dt_ * fr3_husky_model_updater_.qdot_desired_total_;
                    fr3_husky_model_updater_.torque_desired_total_ = fr3_husky_model_updater_.robot_controller_->moveManipulatorJointTorqueStep(fr3_husky_model_updater_.q_desired_total_, fr3_husky_model_updater_.qdot_desired_total_, false);


                    break;
                }
            case 1: // OSF
                {
                    // Build HomePose target for null space (per robot)
                    Eigen::VectorXd HomePose_total(fr3_husky_model_updater_.manipulator_dof_);
                    for(size_t i = 0; i < fr3_husky_model_updater_.num_robots_; ++i) HomePose_total.segment(FR3_DOF*i, FR3_DOF) = HomePose;
                    Eigen::VectorXd null_torque(fr3_husky_model_updater_.robot_data_->getActuatorDof());
                    null_torque.segment(fr3_husky_model_updater_.robot_data_->getActuatorIndex().mani_start, fr3_husky_model_updater_.manipulator_dof_) = 
                        fr3_husky_model_updater_.robot_controller_->moveManipulatorJointTorqueCubic(HomePose_total,
                                                                                            Eigen::VectorXd::Zero(fr3_husky_model_updater_.manipulator_dof_),
                                                                                            q_init_for_home_,
                                                                                            Eigen::VectorXd::Zero(fr3_husky_model_updater_.manipulator_dof_),
                                                                                            time.seconds(),
                                                                                            control_start_time_,
                                                                                            3.0);
                    // Null-space viscous damping: -kd * wheel_vel_ dissipates kinetic energy of the base
                    static constexpr double mobile_null_damping = 10.0; // [N·m·s/rad]: tune as needed
                    null_torque.segment(fr3_husky_model_updater_.robot_data_->getActuatorIndex().mobi_start, fr3_husky_model_updater_.mobile_dof_) =
                        -mobile_null_damping * fr3_husky_model_updater_.wheel_vel_;


                    Eigen::VectorXd wheel_acc_desired = Eigen::VectorXd::Zero(fr3_husky_model_updater_.mobile_dof_);
                    fr3_husky_model_updater_.robot_controller_->OSFStep(ee_data_,
                                                                        wheel_acc_desired,
                                                                        fr3_husky_model_updater_.torque_desired_total_,
                                                                        null_torque);
                    fr3_husky_model_updater_.wheel_vel_desired_ = fr3_husky_model_updater_.wheel_vel_ + wheel_acc_desired * fr3_husky_model_updater_.dt_;
                    break;
                }

            case 2: // QPIK
                {
                    is_qp_solved = fr3_husky_model_updater_.robot_controller_->QPIKStep(ee_data_, qdot_mobile, fr3_husky_model_updater_.qdot_desired_total_, time_verbose);
                    if(!is_qp_solved)
                    {
                        fr3_husky_model_updater_.qdot_desired_total_.setZero();
                        fr3_husky_model_updater_.wheel_vel_desired_.setZero();
                    }
                    fr3_husky_model_updater_.q_desired_total_ = fr3_husky_model_updater_.q_total_ + fr3_husky_model_updater_.dt_ * fr3_husky_model_updater_.qdot_desired_total_;
                    fr3_husky_model_updater_.torque_desired_total_ = fr3_husky_model_updater_.robot_controller_->moveManipulatorJointTorqueStep(fr3_husky_model_updater_.q_desired_total_, fr3_husky_model_updater_.qdot_desired_total_, false);
                    break;
                }


                
                
            case 3: // QPID
                {
                    Eigen::VectorXd wheel_acc_desired = Eigen::VectorXd::Zero(fr3_husky_model_updater_.mobile_dof_);
                    is_qp_solved = fr3_husky_model_updater_.robot_controller_->QPIDStep(ee_data_, wheel_acc_desired, fr3_husky_model_updater_.torque_desired_total_, time_verbose);
                    if(!is_qp_solved)
                    {
                        fr3_husky_model_updater_.torque_desired_total_ = fr3_husky_model_updater_.g_total_;
                        wheel_acc_desired.setZero();
                    }
                    fr3_husky_model_updater_.wheel_vel_desired_ = fr3_husky_model_updater_.wheel_vel_ + wheel_acc_desired * fr3_husky_model_updater_.dt_;
                    break;
                }


                
            default:
                fr3_husky_model_updater_.qdot_desired_total_.setZero();
                fr3_husky_model_updater_.torque_desired_total_.setZero();
                fr3_husky_model_updater_.wheel_vel_desired_.setZero();
                break;
        }

        fr3_husky_model_updater_.writeCommand(fr3_husky_model_updater_.torque_desired_total_ - fr3_husky_model_updater_.g_total_,
            fr3_husky_model_updater_.wheel_vel_desired_);  // robot_controller automatically add gravity force
    
        auto fb = std::make_shared<ActionT::Feedback>();
        fb->is_qp_solved = is_qp_solved;
        fb->time_verbose = time_verbose;
        publishFeedback(fb);

        {
            std::lock_guard<std::mutex> lock(gesture_state_mutex_);
            prev_gesture_states_ = gesture_states_local;
        }
    
        return ComputeResult::RUNNING;
    }

void SAAppleVisionPro::onStop(StopReason reason)
{
    fr3_husky_model_updater_.haltCommands();

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

SAAppleVisionPro::ResultPtr SAAppleVisionPro::makeResult(StopReason reason)
{
    auto result = std::make_shared<ActionT::Result>();
    result->is_completed = true;
    return result;
}


void SAAppleVisionPro::subPoseCallback2(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{   

    Eigen::Vector3d position(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    position = dyros_math::lowPassFilter(position, controller_poses_[0].translation(), 0.001, 0.002);
    Eigen::Quaterniond quaternion(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z);
    quaternion.normalize();
    Eigen::Matrix3d orientation = quaternion.toRotationMatrix();
    {
        std::lock_guard<std::mutex> lock(tracker_pose_mutex_);
        controller_poses_[0].translation() = position;
        controller_poses_[0].linear() = orientation;
    }


}




void SAAppleVisionPro::subPoseCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
{   

    if(msg->poses.size() != NUM_TRACKERS)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Size of PoseArray for tracker_pose (%ld) does not equal to 3.", name_.c_str(), msg->poses.size());
    }
    else
    {
        for(size_t i = 0; i < msg->poses.size(); ++i)
        {
            Eigen::Vector3d position(msg->poses[i].position.x, msg->poses[i].position.y, msg->poses[i].position.z);
            position = dyros_math::lowPassFilter(position, controller_poses_[i].translation(), 0.001, 0.002);
            Eigen::Quaterniond quaternion(msg->poses[i].orientation.w, msg->poses[i].orientation.x, msg->poses[i].orientation.y, msg->poses[i].orientation.z);
            quaternion.normalize();
            Eigen::Matrix3d orientation = quaternion.toRotationMatrix();
            {
                std::lock_guard<std::mutex> lock(tracker_pose_mutex_);
                controller_poses_[i].translation() = position;
                controller_poses_[i].linear() = orientation;
            }

            // pose valid
            tracker_pose_valid_[i] = true;
        }
    }
}

void SAAppleVisionPro::subRightConstraintCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (msg->data.size() != 6)
    {
        RCLCPP_WARN(
            node_->get_logger(),
            "[%s] Size of Float64MultiArray for sa_right_eef_constraint (%ld) does not equal to 6.",
            name_.c_str(),
            msg->data.size());
        return;
    }

    Eigen::Matrix<double, 6, 1> constraint = Eigen::Matrix<double, 6, 1>::Zero();
    for (size_t i = 0; i < 6; ++i)
    {
        constraint(static_cast<Eigen::Index>(i)) = msg->data[i];
    }

    std::lock_guard<std::mutex> lock(right_constraint_mutex_);
    const bool changed = !right_constraint_received_ ||
                         !right_constraint_vector_.isApprox(constraint, 1e-9);
    right_constraint_vector_ = constraint;
    right_constraint_received_ = true;
    if (changed)
    {
        right_constraint_orientation_locked_ = false;
    }
}

void SAAppleVisionPro::subLGestureCallback(const std_msgs::msg::Int32MultiArray::SharedPtr msg)
{
    if(msg->data.size() != NUM_GESTURES)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Size of Int32MultiArray for lhand_gesture (%ld) does not equal to 4.", name_.c_str(), msg->data.size());
    }
    else
    {
        std::lock_guard<std::mutex> lock(gesture_state_mutex_);
        prev_gesture_states_[IDX_LEFT_CON] = gesture_states_[IDX_LEFT_CON];
        for(size_t i = 0; i < msg->data.size(); ++i)
        {
            gesture_states_[IDX_LEFT_CON][i] = (static_cast<int>(msg->data[i]) == 0) ? false : true;
        }

    }
}

void SAAppleVisionPro::subRGestureCallback(const std_msgs::msg::Int32MultiArray::SharedPtr msg)
{
    if(msg->data.size() != NUM_GESTURES)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Size of Int32MultiArray for rhand_gesture (%ld) does not equal to 4.", name_.c_str(), msg->data.size());
    }
    else
    {
        std::lock_guard<std::mutex> lock(gesture_state_mutex_);
        prev_gesture_states_[IDX_RIGHT_CON] = gesture_states_[IDX_RIGHT_CON];
        for(size_t i = 0; i < msg->data.size(); ++i)
        {
            gesture_states_[IDX_RIGHT_CON][i] = (static_cast<int>(msg->data[i]) == 0) ? false : true;
        }

    }
}


Eigen::Affine3d SAAppleVisionPro::smoothAndLimit(const Eigen::Affine3d& prev, const Eigen::Affine3d& target, double dt)
{
    Eigen::Affine3d result = prev;

    // --- 1. Low-pass filter (position)
    Eigen::Vector3d pos =
        (1.0 - smoothing_alpha_) * prev.translation() +
        smoothing_alpha_ * target.translation();

    // --- 2. Velocity limit (position)
    Eigen::Vector3d delta = pos - prev.translation();
    double max_step = max_linear_vel_ * dt;

    if (delta.norm() > max_step) delta = delta.normalized() * max_step;

    result.translation() = prev.translation() + delta;

    // --- 3. Orientation smoothing (slerp)
    Eigen::Quaterniond q_prev(prev.linear());
    Eigen::Quaterniond q_target(target.linear());

    Eigen::Quaterniond q_interp = q_prev.slerp(smoothing_alpha_, q_target);

    // --- 4. Angular velocity limit
    Eigen::AngleAxisd aa(q_prev.inverse() * q_interp);
    double max_angle = max_angular_vel_ * dt;

    if (std::abs(aa.angle()) > max_angle)
    {
        aa.angle() = max_angle;
        q_interp = q_prev * Eigen::Quaterniond(aa);
    }

    result.linear() = q_interp.toRotationMatrix();
    return result;
}


// Register this server into global registry (executed when this TU is linked)
// REGISTER_FR3_ACTION_SERVER(SAAppleVisionPro, "fr3_sa_AVP_tracker")
REGISTER_FR3_HUSKY_ACTION_SERVER(SAAppleVisionPro, "fr3_sa_AVP_tracker")


}  // namespace fr3_husky_controller::servers::fr3
/*
# send goal 
ros2 action send_goal /fr3_sa_AVP_tracker fr3_husky_msgs/action/AppleVisionPro \
"{mode: 1, left_controller_ee_name: 'left_fr3_hand_tcp', right_controller_ee_name: 'right_fr3_hand_tcp', move_orientation: false, controller_pos_multiplier: 1.0, controller_ori_multiplier: 1.0}" \
--feedback
*/
