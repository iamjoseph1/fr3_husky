#include <fr3_husky_controller/servers/fr3/task_move_action_server.hpp>

#include <algorithm>
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

}  // namespace

TaskMove::TaskMove(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater)
: Base(name, node, model_updater),
  fr3_model_updater_(getFR3ModelUpdater(model_updater, name))
{
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
    }

    goal_reached_ = false;
    start_time_ = node_->now().seconds();

    RCLCPP_INFO(node_->get_logger(), "[%s] started for arm='%s'", name_.c_str(), arm_names_.c_str());
}

TaskMove::ComputeResult TaskMove::compute(const rclcpp::Time& time, const rclcpp::Duration& /*period*/)
{
    const double elapsed = time.seconds() - start_time_;
    const double clipped_time = std::min(elapsed, execution_time_);
    const double alpha = execution_time_ > 1e-9 ? std::clamp(clipped_time / execution_time_, 0.0, 1.0) : 1.0;

    if ((arm_names_ == "right" || arm_names_ == "fr3") && hasEE(fr3_model_updater_, "left_fr3_hand_tcp"))
    {
        auto& left_hold = ee_data_["left_fr3_hand_tcp"];
        left_hold = drc::TaskSpaceData::Zero();
        left_hold.x = fr3_model_updater_.robot_data_->getPose("left_fr3_hand_tcp");
        left_hold.xdot = fr3_model_updater_.robot_data_->getVelocity("left_fr3_hand_tcp");
        left_hold.xddot.setZero();
        left_hold.setInit();
        left_hold.x_desired = left_hold.x;
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
        right_hold.x_desired = right_hold.x;
        right_hold.xdot_desired.setZero();
        right_hold.xddot_desired.setZero();
    }

    for (auto& [ee_name, ee_data] : ee_data_)
    {
        ee_data.x = fr3_model_updater_.robot_data_->getPose(ee_name);
        ee_data.xdot = fr3_model_updater_.robot_data_->getVelocity(ee_name);
        ee_data.xddot.setZero();

        const auto target_it = target_poses_.find(ee_name);
        if (target_it == target_poses_.end())
        {
            ee_data.x_desired = ee_data.x_init;
            ee_data.xdot_desired.setZero();
            ee_data.xddot_desired.setZero();
            continue;
        }

        const Eigen::Affine3d& target_pose = target_it->second;

        Eigen::Vector3d p_des;
        for (int i = 0; i < 3; ++i)
        {
            p_des(i) = dyros_math::cubic(
                clipped_time,
                0.0,
                execution_time_,
                ee_data.x_init.translation()(i),
                target_pose.translation()(i),
                0.0,
                0.0);
        }

        ee_data.x_desired = Eigen::Affine3d::Identity();
        ee_data.x_desired.translation() = p_des;
        ee_data.x_desired.linear() = ee_data.x_init.linear();
        ee_data.xdot_desired.setZero();
        ee_data.xddot_desired.setZero();
    }

    fr3_model_updater_.robot_controller_->CLIKStep(
        ee_data_,
        fr3_model_updater_.qdot_desired_total_);

    fr3_model_updater_.q_desired_total_ =
        fr3_model_updater_.q_total_ +
        fr3_model_updater_.dt_ * fr3_model_updater_.qdot_desired_total_;

    fr3_model_updater_.torque_desired_total_ =
        fr3_model_updater_.robot_controller_->moveJointTorqueStep(
            fr3_model_updater_.q_desired_total_,
            fr3_model_updater_.qdot_desired_total_,
            false);

    fr3_model_updater_.writeCommand(
        fr3_model_updater_.torque_desired_total_ - fr3_model_updater_.g_total_);

    auto feedback = std::make_shared<ActionT::Feedback>();
    feedback->progress = alpha;
    publishFeedback(feedback);

    if (elapsed >= execution_time_)
    {
        goal_reached_ = true;
        return ComputeResult::SUCCEEDED;
    }

    return ComputeResult::RUNNING;
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
        result->message = "TaskMove completed";
    }
    else if (reason == StopReason::CANCELED)
    {
        result->message = "TaskMove canceled";
    }
    else
    {
        result->message = "TaskMove aborted";
    }
    return result;
}

REGISTER_FR3_ACTION_SERVER(TaskMove, "fr3_task_move")

}  // namespace fr3_husky_controller::servers::fr3
