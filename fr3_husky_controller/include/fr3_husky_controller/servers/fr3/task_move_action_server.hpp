#pragma once

#include <map>
#include <memory>
#include <future>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <fr3_husky_msgs/action/move_to_joint.hpp>
#include <fr3_husky_msgs/action/task_move.hpp>

#include <fr3_husky_controller/model/fr3_model_updater.hpp>
#include <fr3_husky_controller/servers/action_server_base.hpp>
#include <fr3_husky_controller/utils/dyros_math.h>

namespace fr3_husky_controller::servers::fr3
{

class TaskMove final : public ActionServerBase<fr3_husky_msgs::action::TaskMove>
{
public:
    using ActionT = fr3_husky_msgs::action::TaskMove;
    using Base = ActionServerBase<ActionT>;
    using ComputeResult = typename Base::ComputeResult;
    using StopReason = typename Base::StopReason;
    using ResultPtr = typename Base::ResultPtr;
    using MoveToJointAction = fr3_husky_msgs::action::MoveToJoint;
    using MoveToJointGoalHandle = rclcpp_action::ClientGoalHandle<MoveToJointAction>;

    TaskMove(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater);
    ~TaskMove() override = default;

    int priority() const override { return 8; }
    bool allowPreemption() const override { return false; }

private:
    bool acceptGoal(const ActionT::Goal& goal) override;
    void onGoalAccepted(const ActionT::Goal& goal) override;
    void onStart() override;
    ComputeResult compute(const rclcpp::Time& time, const rclcpp::Duration& period) override;
    void onStop(StopReason reason) override;
    ResultPtr makeResult(StopReason reason) override;

    static Eigen::Affine3d poseStampedToEigen(const geometry_msgs::msg::PoseStamped& pose_msg);
    bool solveIkTarget(Eigen::VectorXd& q_solution, std::string& error_message) const;
    void populateMoveToJointGoal(
        const Eigen::VectorXd& q_solution,
        MoveToJointAction::Goal& goal) const;

    FR3ModelUpdater& fr3_model_updater_;
    rclcpp_action::Client<MoveToJointAction>::SharedPtr move_to_joint_client_;

    std::string arm_names_;
    double execution_time_{3.0};
    double start_time_{0.0};
    std::map<std::string, drc::TaskSpaceData> ee_data_;
    std::map<std::string, Eigen::Affine3d> target_poses_;
    std::map<std::string, Eigen::Vector3d> target_position_deltas_;
    std::map<std::string, bool> use_delta_targets_;
    bool goal_reached_{false};
    std::size_t debug_tick_{0};
    std::shared_future<typename MoveToJointGoalHandle::SharedPtr> move_to_joint_goal_future_;
    bool move_to_joint_goal_pending_{false};
    bool move_to_joint_goal_rejected_{false};
    std::string move_to_joint_dispatch_error_;
};

}  // namespace fr3_husky_controller::servers::fr3
