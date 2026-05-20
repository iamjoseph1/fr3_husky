#include <fr3_husky_controller/servers/fr3/task_move_action_server.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <limits>
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

std::string eeNameFromArmName(const std::string& arm_name)
{
    if (arm_name == "left")
    {
        return "left_fr3_hand_tcp";
    }
    if (arm_name == "right" || arm_name == "fr3")
    {
        return "right_fr3_hand_tcp";
    }
    return "";
}

bool hasEE(const FR3ModelUpdater& model_updater, const std::string& ee_name)
{
    return !ee_name.empty() && model_updater.robot_data_ && model_updater.robot_data_->hasLinkFrame(ee_name);
}

Eigen::Vector3d rotationError(
    const Eigen::Matrix3d& desired_rotation,
    const Eigen::Matrix3d& current_rotation)
{
    const Eigen::Matrix3d rotation_delta = desired_rotation * current_rotation.transpose();
    Eigen::AngleAxisd angle_axis(rotation_delta);
    if (std::abs(angle_axis.angle()) < 1e-9 || angle_axis.axis().squaredNorm() < 1e-12)
    {
        return Eigen::Vector3d::Zero();
    }
    return angle_axis.axis() * angle_axis.angle();
}

}  // namespace

TaskMove::TaskMove(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater)
: Base(name, node, model_updater),
  fr3_model_updater_(getFR3ModelUpdater(model_updater, name))
{
    move_to_joint_client_ = rclcpp_action::create_client<MoveToJointAction>(node_, "fr3_move_to_joint");
    RCLCPP_INFO(node_->get_logger(), "[%s] TaskMove created", name_.c_str());
}

Eigen::Affine3d TaskMove::poseStampedToEigen(const geometry_msgs::msg::PoseStamped& pose_msg)
{
    Eigen::Quaterniond q(
        pose_msg.pose.orientation.w,
        pose_msg.pose.orientation.x,
        pose_msg.pose.orientation.y,
        pose_msg.pose.orientation.z);
    if (q.norm() < 1e-9)
    {
        q = Eigen::Quaterniond::Identity();
    }
    q.normalize();

    Eigen::Affine3d x = Eigen::Affine3d::Identity();
    x.translation() = Eigen::Vector3d(
        pose_msg.pose.position.x,
        pose_msg.pose.position.y,
        pose_msg.pose.position.z);
    x.linear() = q.toRotationMatrix();
    return x;
}

bool TaskMove::acceptGoal(const ActionT::Goal& goal)
{
    if (!model_updater_.HasEffortCommandInterface())
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: effort command interface is required", name_.c_str());
        return false;
    }

    if (goal.execution_time <= 0.0f)
    {
        RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: execution_time must be positive", name_.c_str());
        return false;
    }

    const std::string arm_names = goal.arm_names.empty() ? "right" : goal.arm_names;

    if (arm_names == "right" || arm_names == "left" || arm_names == "fr3")
    {
        if (goal.target_poses.size() != 1)
        {
            RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: single-arm TaskMove requires exactly one target pose", name_.c_str());
            return false;
        }

        const std::string ee_name = eeNameFromArmName(arm_names);
        if (!hasEE(fr3_model_updater_, ee_name))
        {
            RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: EE frame [%s] not available", name_.c_str(), ee_name.c_str());
            return false;
        }
        return true;
    }

    if (arm_names == "both" || arm_names == "dual")
    {
        if (goal.target_poses.size() != 2)
        {
            RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: dual-arm TaskMove requires exactly two target poses", name_.c_str());
            return false;
        }
        if (!hasEE(fr3_model_updater_, "left_fr3_hand_tcp") || !hasEE(fr3_model_updater_, "right_fr3_hand_tcp"))
        {
            RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: dual-arm TaskMove requires both left and right EEs", name_.c_str());
            return false;
        }
        return true;
    }

    RCLCPP_WARN(node_->get_logger(), "[%s] Reject action: arm_names must be one of left, right, fr3, both, dual", name_.c_str());
    return false;
}

void TaskMove::onGoalAccepted(const ActionT::Goal& goal)
{
    arm_names_ = goal.arm_names.empty() ? "right" : goal.arm_names;
    execution_time_ = static_cast<double>(goal.execution_time);
    target_poses_.clear();
    target_position_deltas_.clear();
    use_delta_targets_.clear();

    if (arm_names_ == "both" || arm_names_ == "dual")
    {
        target_poses_["left_fr3_hand_tcp"] = poseStampedToEigen(goal.target_poses[0]);
        target_poses_["right_fr3_hand_tcp"] = poseStampedToEigen(goal.target_poses[1]);
        use_delta_targets_["left_fr3_hand_tcp"] = !goal.abs;
        use_delta_targets_["right_fr3_hand_tcp"] = !goal.abs;
        target_position_deltas_["left_fr3_hand_tcp"] = target_poses_["left_fr3_hand_tcp"].translation();
        target_position_deltas_["right_fr3_hand_tcp"] = target_poses_["right_fr3_hand_tcp"].translation();
    }
    else
    {
        const std::string ee_name = eeNameFromArmName(arm_names_);
        target_poses_[ee_name] = poseStampedToEigen(goal.target_poses[0]);
        use_delta_targets_[ee_name] = !goal.abs;
        target_position_deltas_[ee_name] = target_poses_[ee_name].translation();
    }

    requestActivate();
}

void TaskMove::onStart()
{
    ee_data_.clear();
    for (const auto& [ee_name, target_pose] : target_poses_)
    {
        (void)target_pose;
        ee_data_[ee_name] = drc::TaskSpaceData::Zero();
        ee_data_[ee_name].x = fr3_model_updater_.robot_data_->getPose(ee_name);
        ee_data_[ee_name].xdot = fr3_model_updater_.robot_data_->getVelocity(ee_name);
        ee_data_[ee_name].xddot.setZero();
        ee_data_[ee_name].setInit();
        ee_data_[ee_name].setDesired();

        if (use_delta_targets_[ee_name])
        {
            target_poses_[ee_name] = ee_data_[ee_name].x_init;
            target_poses_[ee_name].translation() += target_position_deltas_[ee_name];
        }

        // Match the old TaskMove semantics: track position only and keep the
        // end-effector orientation fixed at the start pose.
        target_poses_[ee_name].linear() = ee_data_[ee_name].x_init.linear();
    }

    // In dual-arm mode, latch the idle arm pose once so the solver has a
    // fixed hold target instead of chasing the arm's current drifting pose.
    if ((arm_names_ == "right" || arm_names_ == "fr3") && hasEE(fr3_model_updater_, "left_fr3_hand_tcp"))
    {
        auto& left_hold = ee_data_["left_fr3_hand_tcp"];
        left_hold = drc::TaskSpaceData::Zero();
        left_hold.x = fr3_model_updater_.robot_data_->getPose("left_fr3_hand_tcp");
        left_hold.xdot = fr3_model_updater_.robot_data_->getVelocity("left_fr3_hand_tcp");
        left_hold.xddot.setZero();
        left_hold.setInit();
        left_hold.x_desired = left_hold.x_init;
        left_hold.xdot_desired.setZero();
        left_hold.xddot_desired.setZero();
    }
    else if (arm_names_ == "left" && hasEE(fr3_model_updater_, "right_fr3_hand_tcp"))
    {
        auto& right_hold = ee_data_["right_fr3_hand_tcp"];
        right_hold = drc::TaskSpaceData::Zero();
        right_hold.x = fr3_model_updater_.robot_data_->getPose("right_fr3_hand_tcp");
        right_hold.xdot = fr3_model_updater_.robot_data_->getVelocity("right_fr3_hand_tcp");
        right_hold.xddot.setZero();
        right_hold.setInit();
        right_hold.x_desired = right_hold.x_init;
        right_hold.xdot_desired.setZero();
        right_hold.xddot_desired.setZero();
    }

    goal_reached_ = false;
    debug_tick_ = 0;
    start_time_ = node_->now().seconds();
    pending_move_to_joint_goal_ = MoveToJointAction::Goal();
    move_to_joint_goal_ready_ = false;
    move_to_joint_goal_sent_ = false;
    move_to_joint_goal_pending_ = false;
    move_to_joint_goal_rejected_ = false;
    move_to_joint_dispatch_error_.clear();
    move_to_joint_goal_future_ = {};

    if (!move_to_joint_client_->wait_for_action_server(std::chrono::seconds(2)))
    {
        move_to_joint_dispatch_error_ = "fr3_move_to_joint action server is unavailable";
        RCLCPP_ERROR(node_->get_logger(), "[%s] %s", name_.c_str(), move_to_joint_dispatch_error_.c_str());
        return;
    }

    Eigen::VectorXd q_solution;
    std::string ik_error;
    if (!solveIkTarget(q_solution, ik_error))
    {
        move_to_joint_dispatch_error_ = ik_error.empty() ? "IK solve failed" : ik_error;
        RCLCPP_ERROR(node_->get_logger(), "[%s] %s", name_.c_str(), move_to_joint_dispatch_error_.c_str());
        return;
    }

    populateMoveToJointGoal(q_solution, pending_move_to_joint_goal_);
    move_to_joint_goal_ready_ = true;

    RCLCPP_INFO(node_->get_logger(), "[%s] started for arm='%s'", name_.c_str(), arm_names_.c_str());
}

TaskMove::ComputeResult TaskMove::compute(const rclcpp::Time& time, const rclcpp::Duration& /*period*/)
{
    model_updater_.haltCommands();

    auto feedback = std::make_shared<ActionT::Feedback>();
    feedback->progress = 0.0;

    if (!move_to_joint_dispatch_error_.empty())
    {
        publishFeedback(feedback);
        return ComputeResult::ABORTED;
    }

    if (move_to_joint_goal_sent_)
    {
        feedback->progress = 1.0;
        publishFeedback(feedback);
        goal_reached_ = true;
        return ComputeResult::SUCCEEDED;
    }

    if (!move_to_joint_goal_ready_)
    {
        publishFeedback(feedback);
        return ComputeResult::ABORTED;
    }

    auto send_options = rclcpp_action::Client<MoveToJointAction>::SendGoalOptions();
    send_options.goal_response_callback =
        [this](const MoveToJointGoalHandle::SharedPtr& goal_handle)
        {
            if (!goal_handle)
            {
                RCLCPP_ERROR(node_->get_logger(), "[%s] fr3_move_to_joint rejected the goal", name_.c_str());
                return;
            }
            RCLCPP_INFO(node_->get_logger(), "[%s] handed off to fr3_move_to_joint", name_.c_str());
        };
    send_options.result_callback =
        [this](const MoveToJointGoalHandle::WrappedResult& result)
        {
            if (result.code != rclcpp_action::ResultCode::SUCCEEDED)
            {
                RCLCPP_WARN(
                    node_->get_logger(),
                    "[%s] fr3_move_to_joint finished with result code %d",
                    name_.c_str(),
                    static_cast<int>(result.code));
            }
        };

    try
    {
        move_to_joint_client_->async_send_goal(pending_move_to_joint_goal_, send_options);
        move_to_joint_goal_sent_ = true;
        move_to_joint_goal_ready_ = false;
        feedback->progress = 1.0;
        publishFeedback(feedback);
        goal_reached_ = true;
        return ComputeResult::SUCCEEDED;
    }
    catch (const std::exception& e)
    {
        move_to_joint_dispatch_error_ = std::string("failed to send fr3_move_to_joint goal: ") + e.what();
        publishFeedback(feedback);
        return ComputeResult::ABORTED;
    }

    publishFeedback(feedback);
    return ComputeResult::ABORTED;
}

void TaskMove::onStop(StopReason reason)
{
    model_updater_.haltCommands();

    const char* reason_str = "none";
    if (reason == StopReason::CANCELED) reason_str = "canceled";
    else if (reason == StopReason::SUCCEEDED) reason_str = "succeeded";
    else if (reason == StopReason::ABORTED) reason_str = "aborted";

    RCLCPP_INFO(node_->get_logger(), "[%s] stopped (%s)", name_.c_str(), reason_str);
}

TaskMove::ResultPtr TaskMove::makeResult(StopReason reason)
{
    auto result = std::make_shared<ActionT::Result>();
    result->success = (reason == StopReason::SUCCEEDED);
    if (reason == StopReason::SUCCEEDED)
    {
        result->message = "TaskMove handed off to fr3_move_to_joint";
    }
    else if (reason == StopReason::CANCELED)
    {
        result->message = "TaskMove canceled";
    }
    else
    {
        result->message = move_to_joint_dispatch_error_.empty() ? "TaskMove aborted" : move_to_joint_dispatch_error_;
    }
    return result;
}

bool TaskMove::solveIkTarget(Eigen::VectorXd& q_solution, std::string& error_message) const
{
    if (!fr3_model_updater_.robot_data_)
    {
        error_message = "robot_data is unavailable";
        return false;
    }

    q_solution = fr3_model_updater_.q_total_;
    const Eigen::Index dof = q_solution.size();
    if (dof <= 0)
    {
        error_message = "invalid manipulator DoF";
        return false;
    }

    std::vector<std::pair<std::string, Eigen::Affine3d>> tasks;
    tasks.reserve(target_poses_.size() + 1);
    for (const auto& [ee_name, target_pose] : target_poses_)
    {
        tasks.emplace_back(ee_name, target_pose);
    }

    if (fr3_model_updater_.num_robots_ == 2)
    {
        if ((arm_names_ == "right" || arm_names_ == "fr3") && ee_data_.count("left_fr3_hand_tcp") > 0)
        {
            tasks.emplace_back("left_fr3_hand_tcp", ee_data_.at("left_fr3_hand_tcp").x_init);
        }
        else if (arm_names_ == "left" && ee_data_.count("right_fr3_hand_tcp") > 0)
        {
            tasks.emplace_back("right_fr3_hand_tcp", ee_data_.at("right_fr3_hand_tcp").x_init);
        }
    }

    if (tasks.empty())
    {
        error_message = "no IK tasks to solve";
        return false;
    }

    constexpr int kMaxIterations = 200;
    constexpr double kStepScale = 0.4;
    constexpr double kDamping = 1e-3;
    constexpr double kPositionTolerance = 1e-3;
    constexpr double kOrientationTolerance = 5e-3;
    constexpr double kMaxDeltaNorm = 0.2;

    double last_error_norm = std::numeric_limits<double>::infinity();
    for (int iter = 0; iter < kMaxIterations; ++iter)
    {
        const Eigen::Index task_dim = static_cast<Eigen::Index>(6 * tasks.size());
        Eigen::VectorXd error = Eigen::VectorXd::Zero(task_dim);
        Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(task_dim, dof);

        double max_pos_err = 0.0;
        double max_rot_err = 0.0;
        for (std::size_t task_index = 0; task_index < tasks.size(); ++task_index)
        {
            const auto& [ee_name, target_pose] = tasks[task_index];
            const Eigen::Affine3d current_pose = fr3_model_updater_.robot_data_->computePose(q_solution, ee_name);
            const Eigen::MatrixXd jacobian_full = fr3_model_updater_.robot_data_->computeJacobian(q_solution, ee_name);

            const Eigen::Vector3d pos_err = target_pose.translation() - current_pose.translation();
            const Eigen::Vector3d rot_err = rotationError(target_pose.linear(), current_pose.linear());

            max_pos_err = std::max(max_pos_err, pos_err.norm());
            max_rot_err = std::max(max_rot_err, rot_err.norm());

            const Eigen::Index row = static_cast<Eigen::Index>(6 * task_index);
            error.segment<3>(row) = pos_err;
            error.segment<3>(row + 3) = rot_err;
            jacobian.block(row, 0, 6, dof) = jacobian_full.leftCols(dof);
        }

        if (max_pos_err < kPositionTolerance && max_rot_err < kOrientationTolerance)
        {
            return true;
        }

        Eigen::MatrixXd normal = jacobian * jacobian.transpose();
        normal.diagonal().array() += kDamping * kDamping;
        Eigen::VectorXd delta_q = jacobian.transpose() * normal.ldlt().solve(error);

        if (fr3_model_updater_.num_robots_ == 2)
        {
            if (arm_names_ == "right" || arm_names_ == "fr3")
            {
                delta_q.segment(0, FR3_DOF).setZero();
            }
            else if (arm_names_ == "left")
            {
                delta_q.segment(FR3_DOF, FR3_DOF).setZero();
            }
        }

        const double delta_norm = delta_q.norm();
        if (!std::isfinite(delta_norm))
        {
            error_message = "IK produced non-finite joint update";
            return false;
        }
        if (delta_norm > kMaxDeltaNorm)
        {
            delta_q *= (kMaxDeltaNorm / delta_norm);
        }

        q_solution += kStepScale * delta_q;
        last_error_norm = error.norm();
    }

    error_message = "IK did not converge; final error norm=" + std::to_string(last_error_norm);
    return false;
}

void TaskMove::populateMoveToJointGoal(
    const Eigen::VectorXd& q_solution,
    MoveToJointAction::Goal& goal) const
{
    goal.joint_names.clear();
    goal.target_positions.clear();
    goal.max_velocity_scaling_factor = 0.1;
    goal.max_acceleration_scaling_factor = 0.1;

    const auto append_arm = [&goal, &q_solution](const std::string& robot_name, Eigen::Index offset)
    {
        for (int joint_index = 0; joint_index < FR3_DOF; ++joint_index)
        {
            goal.joint_names.push_back(
                robot_name + "_fr3_joint" + std::to_string(joint_index + 1));
            goal.target_positions.push_back(q_solution(offset + joint_index));
        }
    };

    if (arm_names_ == "both" || arm_names_ == "dual")
    {
        append_arm("left", 0);
        append_arm("right", FR3_DOF);
    }
    else if (arm_names_ == "left")
    {
        append_arm("left", 0);
    }
    else
    {
        const Eigen::Index offset = (fr3_model_updater_.num_robots_ == 2) ? FR3_DOF : 0;
        append_arm("right", offset);
    }
}

REGISTER_FR3_ACTION_SERVER(TaskMove, "fr3_task_move")

}  // namespace fr3_husky_controller::servers::fr3

//  ros2 action send_goal /fr3_task_move fr3_husky_msgs/action/TaskMove "{arm_names: 'right', target_poses: [{header: {frame_id: ''}, pose: {position: {x: 0.05, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}], execution_time: 3.0, abs: false}"
