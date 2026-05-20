#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

#include <action_msgs/msg/goal_status.hpp>
#include <action_msgs/msg/goal_status_array.hpp>
#include <fr3_husky_msgs/action/move_to_joint.hpp>
#include <fr3_husky_msgs/action/vive_tracker.hpp>

#include <fr3_husky_controller/servers/action_server_base.hpp>
#include <fr3_husky_controller/model/fr3_model_updater.hpp>
#include <fr3_husky_controller/utils/dyros_math.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <sensor_msgs/msg/joy.hpp>

#define NUM_TRACKERS    3 // left, right, head
#define NUM_CONTROLLERS 2 // left, right (only Focus3 controllers)
#define IDX_LEFT_CON    0 // index of left controller pose in "tracker_pose" topic
#define IDX_RIGHT_CON   1 // index of right controller pose in "tracker_pose" topic
#define IDX_HEAD_CON    2 // index of HMD head pose in "tracker_pose" topic

#define NUM_BUTTONS        5 // number of buttons in Focu3 controller [trigger, grip, a, b, joy]
#define IDX_TRIGGER_BUTTON 0 // index of Trigger button in "l/rhand_joy" topic
#define IDX_GRIP_BUTTON    1 // index of Grip button in "l/rhand_joy" topic
#define IDX_A_BUTTON       2 // index of A button in "l/rhand_joy" topic
#define IDX_B_BUTTON       3 // index of B button in "l/rhand_joy" topic
#define IDX_JOY_BUTTON     4 // index of Joy button in "l/rhand_joy" topic

namespace fr3_husky_controller::servers::fr3
{

class SAViveTracker final : public ActionServerBase<fr3_husky_msgs::action::ViveTracker>
{
public:
    using ActionT = fr3_husky_msgs::action::ViveTracker;
    using Base = ActionServerBase<ActionT>;
    using ComputeResult = typename Base::ComputeResult;
    using StopReason = typename Base::StopReason;
    using ResultPtr = typename Base::ResultPtr;

    SAViveTracker(const std::string& name, const NodePtr& node, ModelUpdaterBase& model_updater);
    ~SAViveTracker() override;

    int priority() const override { return 7; }
    bool allowPreemption() const override { return true; }

private:
    bool acceptGoal(const ActionT::Goal& goal) override;
    void onGoalAccepted(const ActionT::Goal& goal) override;
    void onStart() override;
    ComputeResult compute(const rclcpp::Time& time, const rclcpp::Duration& period) override;
    void onStop(StopReason reason) override;
    ResultPtr makeResult(StopReason reason) override;

private:
    FR3ModelUpdater& fr3_model_updater_;

private:
    // publishers & subscribers
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr  pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr l_joy_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr r_joy_sub_;
    rclcpp::TimerBase::SharedPtr right_constraint_udp_poll_timer_;

    void subPoseCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg);
    void subLJoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg);
    void subRJoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg);
    void pollRightConstraintUdp();
    bool handleRightConstraintVector(const Eigen::Vector3d& constraint);
    bool parseRightConstraintUdpPayload(const uint8_t* data, size_t size, Eigen::Vector3d& constraint) const;
    bool sendFrontOverviewTriggerUdp();

    // vive controller state data
    std::vector<Eigen::Affine3d> controller_poses_;      // left, right, head
    std::vector<Eigen::Affine3d> controller_poses_init_; // left, right, head
    std::vector<std::vector<bool>> button_states_;       // [left, right][trigger, grip, a, b, joy]
    std::vector<std::vector<bool>> prev_button_states_;  // previous button states for edge detection

    // states for ...
    std::vector<bool> is_mouse_mode_on_{false, false};
    bool is_initialize_mode_on_{false};
    std::vector<bool> is_gripper_mode_on_{false, false};
    
    // robot data
    std::vector<Eigen::Matrix3d> tracker_base2robot_base_;
    std::map<std::string, drc::TaskSpaceData> ee_data_;
    bool has_mobile_{false};

    // action goal data
    int control_mode_;                     // 0: CLIK, 1: OSF, 2:QPIK, 3:QPID
    std::string left_controller_ee_name_;  // EE name for tracking left vive controller
    std::string right_controller_ee_name_; // EE name for tracking right vive controller
    bool move_ori_;
    double controller_pos_multiplier_;
    double controller_ori_multiplier_;

    std::mutex tracker_pose_mutex_;
    std::mutex button_state_mutex_;
    std::mutex right_constraint_mutex_;

    Eigen::Vector3d right_constraint_vector_ = Eigen::Vector3d::Zero();
    bool right_constraint_received_{false};
    bool right_constraint_applying_{false};
    bool right_constraint_orientation_locked_{false};
    bool right_constraint_anchor_pose_locked_{false};
    Eigen::Affine3d right_constraint_anchor_pose_ = Eigen::Affine3d::Identity();
    Eigen::Matrix3d right_constraint_locked_orientation_ = Eigen::Matrix3d::Identity();
    int front_overview_trigger_udp_sock_{-1};
    int right_constraint_udp_sock_{-1};
    std::string front_overview_trigger_udp_ip_{"100.83.23.26"}; // 5090v1 서버 tailscale IP
    int64_t front_overview_trigger_udp_port_{5008};
    bool front_overview_trigger_udp_value_{true};
    std::string right_constraint_udp_bind_ip_{"0.0.0.0"};
    int64_t right_constraint_udp_bind_port_{5009};

    // initialize mode: button A -> send goal to fr3_move_to_joint
    const Eigen::Vector<double, FR3_DOF> HomePose{0., -0.785, 0.0, -2.356, 0.0, 1.571, 0.785};
    ActionT::Goal saved_vive_goal_{};  // saved goal params for auto-resume after init

    // action clients
    using MoveToJointAction = fr3_husky_msgs::action::MoveToJoint;
    rclcpp_action::Client<MoveToJointAction>::SharedPtr move_to_joint_client_;
    rclcpp_action::Client<ActionT>::SharedPtr           vt_self_client_;  // self-client for resume

    // JTC completion monitoring: wait for JTC to finish executing before re-activating
    rclcpp::Subscription<action_msgs::msg::GoalStatusArray>::SharedPtr jtc_status_sub_;
    std::atomic<bool> waiting_for_jtc_{false};
};

}  // namespace fr3_husky_controller::servers::fr3
