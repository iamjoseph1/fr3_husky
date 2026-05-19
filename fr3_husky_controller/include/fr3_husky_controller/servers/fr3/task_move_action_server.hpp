#pragma once

#include <map>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>

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

    FR3ModelUpdater& fr3_model_updater_;

    std::string arm_names_;
    double execution_time_{3.0};
    double start_time_{0.0};
    std::map<std::string, drc::TaskSpaceData> ee_data_;
    std::map<std::string, Eigen::Affine3d> target_poses_;
    std::map<std::string, Eigen::Vector3d> target_position_deltas_;
    std::map<std::string, bool> use_delta_targets_;
    bool goal_reached_{false};
};

}  // namespace fr3_husky_controller::servers::fr3
