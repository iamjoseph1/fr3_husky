#include "fr3_husky_controller/fr3_husky_action_controller.hpp"

#include <mujoco/mujoco.h>
#include <unordered_set>
#include <random>
#include <controller_manager_msgs/srv/list_hardware_interfaces.hpp>

// --------------- MuJoco object random spawn ---------------
namespace mujoco_ros_hardware
{
class MujocoWorldSingleton
{
public:
    static MujocoWorldSingleton& get();
    bool isSceneLoaded() const;
    const std::string& xacroPath() const;
    mjModel* model() const;
    mjData* data() const;
    std::mutex& dataMutex();
};
}  // namespace mujoco_ros_hardware

// --------------- MuJoco object random spawn ---------------

namespace fr3_husky_controller
{
namespace
{
    constexpr const char* kJoyTopic = "/joy";

// --------------- MuJoco object random spawn ---------------
    constexpr double kSceneSpawnPerturbRadiusM = 0.0; // 학습시 0.1
    constexpr double kSceneSpawnMaxYawDeg = 30.0; // 학습시 20.0
    constexpr double kTwoPi = 6.28318530717958647692;

Eigen::Vector2d sampleDiskOffset(double radius_m)
{
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> angle_dist(0.0, kTwoPi);
    std::uniform_real_distribution<double> unit_dist(0.0, 1.0);

    const double angle = angle_dist(rng);
    const double radius = radius_m * std::sqrt(unit_dist(rng));
    return Eigen::Vector2d(radius * std::cos(angle), radius * std::sin(angle));
}

bool perturbStaticBodyXY(mjModel* model, const std::string& body_name, const Eigen::Vector2d& xy_offset)
{
    const int body_id = mj_name2id(model, mjOBJ_BODY, body_name.c_str());
    if (body_id < 0) return false;

    model->body_pos[3 * body_id + 0] += xy_offset.x();
    model->body_pos[3 * body_id + 1] += xy_offset.y();
    return true;
}

double sampleYawRadiansFromDegreeLimit(double max_abs_deg)
{
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> yaw_deg_dist(max_abs_deg*0.7, max_abs_deg);
    return yaw_deg_dist(rng) * M_PI / 180.0;
}

bool applyStaticBodyYawWorldZ(mjModel* model, const std::string& body_name, double yaw_rad)
{
    const int body_id = mj_name2id(model, mjOBJ_BODY, body_name.c_str());
    if (body_id < 0) return false;

    Eigen::Quaterniond q_body(
        model->body_quat[4 * body_id + 0],
        model->body_quat[4 * body_id + 1],
        model->body_quat[4 * body_id + 2],
        model->body_quat[4 * body_id + 3]);
    const Eigen::Quaterniond q_yaw(Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ()));
    q_body = (q_yaw * q_body).normalized();

    model->body_quat[4 * body_id + 0] = q_body.w();
    model->body_quat[4 * body_id + 1] = q_body.x();
    model->body_quat[4 * body_id + 2] = q_body.y();
    model->body_quat[4 * body_id + 3] = q_body.z();
    return true;
}

bool perturbFreeJointXY(const mjModel* model, mjData* data, const std::string& joint_name, const Eigen::Vector2d& xy_offset)
{
    const int joint_id = mj_name2id(model, mjOBJ_JOINT, joint_name.c_str());
    if (joint_id < 0) return false;

    const int qpos_adr = model->jnt_qposadr[joint_id];
    data->qpos[qpos_adr + 0] += xy_offset.x();
    data->qpos[qpos_adr + 1] += xy_offset.y();
    return true;
}

bool applyFreeJointYawWorldZ(const mjModel* model, mjData* data, const std::string& joint_name, double yaw_rad)
{
    const int joint_id = mj_name2id(model, mjOBJ_JOINT, joint_name.c_str());
    if (joint_id < 0) return false;

    const int qpos_adr = model->jnt_qposadr[joint_id];
    Eigen::Quaterniond q_body(
        data->qpos[qpos_adr + 3],
        data->qpos[qpos_adr + 4],
        data->qpos[qpos_adr + 5],
        data->qpos[qpos_adr + 6]);
    const Eigen::Quaterniond q_yaw(Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ()));
    q_body = (q_yaw * q_body).normalized();

    data->qpos[qpos_adr + 3] = q_body.w();
    data->qpos[qpos_adr + 4] = q_body.x();
    data->qpos[qpos_adr + 5] = q_body.y();
    data->qpos[qpos_adr + 6] = q_body.z();
    return true;
}

void maybeRandomizeSceneSpawn(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr& node, bool& randomized_flag)
{
    if (randomized_flag) return;

    auto& world = mujoco_ros_hardware::MujocoWorldSingleton::get();
    if (!world.isSceneLoaded()) return;

    const std::string xacro_path = world.xacroPath();
    if (xacro_path.empty()) return;

    std::lock_guard<std::mutex> lock(world.dataMutex());
    mjModel* model = world.model();
    mjData* data = world.data();
    if (!model || !data) return;

    const Eigen::Vector2d xy_offset = sampleDiskOffset(kSceneSpawnPerturbRadiusM);
    bool changed = false;
    double applied_yaw_deg = 0.0;

    if (xacro_path.find("dual_fr3_husky_threading.xml.xacro") != std::string::npos)
    {
        constexpr double kThreadingSceneSpawnMaxYawDeg = 0.0;
        const double yaw_rad = sampleYawRadiansFromDegreeLimit(kThreadingSceneSpawnMaxYawDeg);
        applied_yaw_deg = yaw_rad * 180.0 / M_PI;
        changed = perturbStaticBodyXY(model, "tripod_obj", xy_offset);
        changed = applyStaticBodyYawWorldZ(model, "tripod_obj", yaw_rad) || changed;
    }
    else if (xacro_path.find("dual_fr3_husky_threepieceassembly.xml.xacro") != std::string::npos)
    {
        const double yaw_rad = sampleYawRadiansFromDegreeLimit(kSceneSpawnMaxYawDeg);
        applied_yaw_deg = yaw_rad * 180.0 / M_PI;
        const bool base_ok = perturbFreeJointXY(model, data, "base_joint", xy_offset);
        const bool piece_ok = perturbFreeJointXY(model, data, "piece_1_joint", xy_offset);
        const bool base_yaw_ok = applyFreeJointYawWorldZ(model, data, "base_joint", yaw_rad);
        const bool piece_yaw_ok = applyFreeJointYawWorldZ(model, data, "piece_1_joint", yaw_rad);
        changed = base_ok || piece_ok || base_yaw_ok || piece_yaw_ok;
    }
    else if (xacro_path.find("dual_fr3_husky_square.xml.xacro") != std::string::npos)
    {
        const double yaw_rad = sampleYawRadiansFromDegreeLimit(kSceneSpawnMaxYawDeg);
        applied_yaw_deg = 0.5 * yaw_rad * 180.0 / M_PI;
        changed = perturbStaticBodyXY(model, "peg1", xy_offset);
        changed = applyStaticBodyYawWorldZ(model, "peg1", yaw_rad) || changed;
    }
    else if (xacro_path.find("dual_fr3_husky_coffee.xml.xacro") != std::string::npos)
    {
        const double yaw_rad = sampleYawRadiansFromDegreeLimit(kSceneSpawnMaxYawDeg);
        applied_yaw_deg = yaw_rad * 180.0 / M_PI;
        changed = perturbStaticBodyXY(model, "coffee_machine_root", xy_offset);
        changed = applyStaticBodyYawWorldZ(model, "coffee_machine_root", yaw_rad) || changed;
    }

    if (!changed) return;

    mj_forward(model, data);
    randomized_flag = true;
    RCLCPP_INFO(
        node->get_logger(),
        "[SceneSpawn] Applied XY/yaw perturbation for scene '%s': radius<=%.3f m max_yaw=%.1f deg dx=%.4f dy=%.4f yaw_deg=%.2f",
        xacro_path.c_str(),
        kSceneSpawnPerturbRadiusM,
        kSceneSpawnMaxYawDeg,
        xy_offset.x(),
        xy_offset.y(),
        applied_yaw_deg);
}

// --------------- MuJoco object random spawn ---------------

}  // namespace

controller_interface::InterfaceConfiguration FR3HuskyActionController::state_interface_configuration() const
{
    controller_interface::InterfaceConfiguration conf;
    conf.type = controller_interface::interface_configuration_type::INDIVIDUAL;

    // Manipulator joint state interfaces
    for (const auto & joint_name : params_.manipulator_joints)
    {
        for (const auto & interface_type : params_.manipulator_state_interfaces)
        {
            conf.names.push_back(joint_name + "/" + interface_type);
        }
    }

    // Franka semantic components (model/state) — only when HW provides them
    if (use_franka_model_)
    {
        for (size_t i = 0; i < num_robots_; ++i)
        {
            for (const auto & name : franka_robot_model_[i]->get_state_interface_names())
            {
                conf.names.push_back(name);
            }
            conf.names.push_back(params_.robot_name[i] + "_" + arm_id_ + "/robot_time");
        }
    }

    // Mobile base state interfaces (position/velocity)
    for (const auto & joint_name : params_.left_wheel_names)
    {
        conf.names.push_back(joint_name + "/" + allowed_interface_types_[0]);
        conf.names.push_back(joint_name + "/" + allowed_interface_types_[1]);
    }
    for (const auto & joint_name : params_.right_wheel_names)
    {
        conf.names.push_back(joint_name + "/" + allowed_interface_types_[0]);
        conf.names.push_back(joint_name + "/" + allowed_interface_types_[1]);
    }

    return conf;
}

controller_interface::InterfaceConfiguration FR3HuskyActionController::command_interface_configuration() const
{
    controller_interface::InterfaceConfiguration conf;
    conf.type = controller_interface::interface_configuration_type::INDIVIDUAL;

    // Manipulator command interfaces
    for (const auto & joint_name : params_.manipulator_joints)
    {
        conf.names.push_back(joint_name + "/" + params_.manipulator_command_interface);
    }

    // Mobile base command interfaces (velocity)
    for (const auto & joint_name : params_.left_wheel_names)
    {
        conf.names.push_back(joint_name + "/" + allowed_interface_types_[1]);
    }
    for (const auto & joint_name : params_.right_wheel_names)
    {
        conf.names.push_back(joint_name + "/" + allowed_interface_types_[1]);
    }

    return conf;
}

CallbackReturn FR3HuskyActionController::on_init()
{
    try
    {
        // Create the parameter listener and get the parameters
        param_listener_ = std::make_shared<ParamListener>(get_node());
        params_ = param_listener_->get_params();
    }
    catch (const std::exception & e)
    {
        fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
        return CallbackReturn::ERROR;
    }
    return CallbackReturn::SUCCESS;
}



CallbackReturn FR3HuskyActionController::on_configure(const rclcpp_lifecycle::State & /*previous_state*/)
{
    

    // update parameters if they have changed
    if (param_listener_->is_old(params_))
    {
        params_ = param_listener_->get_params();
        LOGI(get_node(), "Parameters were updated");
    }


    // number of FR3 robot used
    num_robots_ = params_.robot_name.size();
    if (num_robots_ < 1 || num_robots_ > 2)
    {
        LOGE(get_node(), "FR3 Husky controller expects one or two FR3 robots, but got %zu.", num_robots_);
        return CallbackReturn::FAILURE;
    }


    // check names in allowed name list & uniqueness 
    const std::unordered_set<std::string> allowed = {"left", "right"};
    std::unordered_set<std::string> seen;
    for (const auto& name : params_.robot_name) 
    {
        if (allowed.find(name) == allowed.end()) 
        {
            LOGE(get_node(), "Invalid robot_name '%s'. Allowed values are 'left' or 'right'.", name.c_str());
            return CallbackReturn::FAILURE;
        }

        if (!seen.insert(name).second) 
        {
            LOGE(get_node(), "Duplicate robot_name '%s' detected. robot_name entries must be unique.", name.c_str());
            return CallbackReturn::FAILURE;
        }
    }


    // get manipulator degrees of freedom
    manipulator_dof_ = params_.manipulator_joints.size();
    if (manipulator_dof_ != FR3_DOF * num_robots_)
    {
        LOGE(get_node(), "FR3 Husky controller expects %zu manipulator DoF, but got %zu.", FR3_DOF * num_robots_, manipulator_dof_);
        return CallbackReturn::FAILURE;
    }


    // number of mobile wheel checking
    if (params_.left_wheel_names.size() != params_.right_wheel_names.size())
    {
        LOGE(get_node(), "Number of left wheels (%zu) differs from right wheels (%zu).", params_.left_wheel_names.size(), params_.right_wheel_names.size());
        return CallbackReturn::FAILURE;
    }
    if (params_.left_wheel_names.size() != WHEEL_PER_SIDE)
    {
        LOGW(get_node(), "Expected %d wheels per side; got %zu. Continuing but commands will be duplicated.", WHEEL_PER_SIDE, params_.left_wheel_names.size());
    }

    if (params_.manipulator_command_interface.empty() || params_.manipulator_state_interfaces.empty())
    {
        LOGE(get_node(), "Command/state interface parameters are empty.");
        return CallbackReturn::FAILURE;
    }
    if (params_.estop_button_index < 0)
    {
        LOGE(get_node(), "estop_button_index must be non-negative, got %ld", static_cast<long>(params_.estop_button_index));
        return CallbackReturn::FAILURE;
    }


    // command flags (exactly one will be true)
    has_position_command_interface_ = (params_.manipulator_command_interface == allowed_interface_types_[0]);
    has_velocity_command_interface_ = (params_.manipulator_command_interface == allowed_interface_types_[1]);
    has_effort_command_interface_   = (params_.manipulator_command_interface == allowed_interface_types_[2]);

    has_position_state_interface_ = robot_utils::contains_interface_type(params_.manipulator_state_interfaces, allowed_interface_types_[0]);
    has_velocity_state_interface_ = robot_utils::contains_interface_type(params_.manipulator_state_interfaces, allowed_interface_types_[1]);
    has_effort_state_interface_   = robot_utils::contains_interface_type(params_.manipulator_state_interfaces, allowed_interface_types_[2]);

    // Validation of combinations of state and velocity together have to be done
    // here because the parameter validators only deal with each parameter
    // separately.
    if (has_velocity_command_interface_ && (!has_velocity_state_interface_ || !has_position_state_interface_))
    {
        LOGE(get_node(),
            "'velocity' command interface can only be used alone if 'velocity' and "
            "'position' state interfaces are present");
        return CallbackReturn::FAILURE;
    }

    // effort interfaces require position and velocity state
    if (has_effort_command_interface_ && (!has_velocity_state_interface_ || !has_position_state_interface_))
    {
        LOGE(get_node(),
                "'effort' command interface can only be used alone or with 'position' command interface "
                "if 'velocity' and 'position' state interfaces are present");
        return CallbackReturn::FAILURE;
    }

    // get sampling time dt_
    if (get_update_rate() == 0)
    {
        throw std::runtime_error("Controller's update rate is set to 0. This should not happen!");
    }
    dt_ = 1.0 / static_cast<double>(get_update_rate());


    if (!model_updater_)
    {
        model_updater_ = std::make_unique<FR3HuskyModelUpdater>();
    }


    publish_rate_ = params_.publish_rate;


    joy_msg_received_.store(false, std::memory_order_release);
    estop_button_pressed_.store(false, std::memory_order_release);
    estop_is_active_ = false;
    estop_button_index_warned_ = false;
    joy_subscriber_ = get_node()->create_subscription<sensor_msgs::msg::Joy>(
        kJoyTopic, rclcpp::SystemDefaultsQoS(),
        std::bind(&FR3HuskyActionController::onJoyMessage, this, std::placeholders::_1));



    heavy_init_done_ = false;

    if (initialize_heavy_resources() != CallbackReturn::SUCCESS)
    {
        LOGE(get_node(), "Heavy initialization failed during configure.");
        return CallbackReturn::ERROR;
    }

    return CallbackReturn::SUCCESS;

}

CallbackReturn FR3HuskyActionController::initialize_heavy_resources()
{
    std::scoped_lock<std::mutex> lock(heavy_init_mutex_);
    if (heavy_init_done_)
    {
        return CallbackReturn::SUCCESS;
    }


    // for finding whether hand/mobile base exist
    auto tmp_node = rclcpp::Node::make_shared("_tmp_urdf_client_" + std::string(get_node()->get_name()));
    auto param_client = std::make_shared<rclcpp::SyncParametersClient>(tmp_node, "controller_manager");
    param_client->wait_for_service();
    auto cm_params = param_client->get_parameters({"robot_description"});
    std::string tmp_urdf_xml = cm_params[0].value_to_string();

    /// When using MuJoCo, do not use franka sementic segment
    use_franka_model_ = (tmp_urdf_xml.find("mujoco_ros_hardware/MujocoHardwareInterface") == std::string::npos);
    LOGW(get_node(), "Franka model: %s", use_franka_model_ ? "available" : "unavailable (pinocchio fallback)");

    // Initialize Franka semantic components (only if HW exports robot_model)
    franka_robot_model_.clear();
    if (use_franka_model_)
    {
        for (const auto & name : params_.robot_name)
        {
            franka_robot_model_.push_back(std::make_unique<franka_semantic_components::FrankaRobotModel>(name + "_" + arm_id_ + "/robot_model",
                                                                                                         name + "_" + arm_id_ + "/robot_state"));
        }
    }


    pinocchio::Model pin_model;
    pinocchio::urdf::buildModelFromXML(tmp_urdf_xml, pin_model);
    pinocchio::Data pin_data = pinocchio::Data(pin_model);

    bool has_hand = false;

    for (const auto& frame : pin_model.frames)
    {
        if (frame.name.find("hand") != std::string::npos) has_hand = true;
        if (has_hand) break;
    }


    // initialize dyros_robot_data & controller
    const std::string description_pkg = ament_index_cpp::get_package_share_directory("fr3_husky_description");
    std::string xacro_path = description_pkg + "/robots/";
    std::ostringstream segmentation_args;
    segmentation_args << " with_sc:=true"
                      << " hand:=" << (has_hand ? "true" : "false")
                      << " as_two_wheels:=true"
                      << " ros2_control:=false"
                      << " use_fake_hardware:=false"
                      << " fake_sensor_commands:=false"
                      //   << " virtual_joint:=true";
                      << " virtual_joint:=false";
    std::string robot_segmentation_description_param = segmentation_args.str();
    std::string robot_description_param = robot_segmentation_description_param + " ros2_control:=false"
                                                                               + " use_fake_hardware:=false"
                                                                               + " fake_sensor_commands:=false"
                                                                               + " fix_finger:=true"
                                                                               + " virtual_joint:=true";

    if (num_robots_ == 1)
    {
        robot_segmentation_description_param += " side:=" + params_.robot_name[0];
        robot_description_param += " side:=" + params_.robot_name[0];
        xacro_path += "single_fr3_husky";
    }
    else
    {
        xacro_path += "dual_fr3_husky";
    }

    const std::string urdf_xml = robot_utils::execAndCaptureStdout("xacro " + xacro_path + ".urdf.xacro" + robot_description_param);
    const std::string srdf_xml = robot_utils::execAndCaptureStdout("xacro " + xacro_path + ".srdf.xacro" + robot_segmentation_description_param);

    drc::Mobile::KinematicParam p;
    p.type          = drc::Mobile::DriveType::Differential;
    p.wheel_radius  = params_.wheel_radius_multiplier * params_.wheel_radius;
    p.base_width    = params_.wheel_separation_multiplier * params_.wheel_separation;
    p.max_lin_speed = params_.linear.x.max_velocity;
    p.max_ang_speed = params_.angular.z.max_velocity;
    p.max_lin_acc   = params_.linear.x.max_acceleration;
    p.max_ang_acc   = params_.angular.z.max_acceleration;

    // check the indices of URDF by urdf_to_graphviz
    // its order depends on DFS from root link
    drc::MobileManipulator::JointIndex j;
    if(!setJointIndex(urdf_xml, j))
    {
        LOGE(get_node(), "Failed to set mobile manipulator JointIndex");
        return CallbackReturn::ERROR;
    }



    drc::MobileManipulator::ActuatorIndex a;
    a.mani_start = j.mani_start - 3;
    a.mobi_start = j.mobi_start - 3;

    std::shared_ptr<drc::MobileManipulator::RobotData> robot_data;
    try
    {
        robot_data = std::make_shared<drc::MobileManipulator::RobotData>(dt_, p, j, a, urdf_xml, srdf_xml, description_pkg, true);

        LOGI(get_node(), robot_data->getVerbose().c_str());
    }
    catch (const std::exception & e)
    {
        LOGE(get_node(), "Failed to initialize RobotData: %s", e.what());
        return CallbackReturn::ERROR;
    }


    std::shared_ptr<drc::MobileManipulator::RobotController> robot_controller = std::make_shared<drc::MobileManipulator::RobotController>(robot_data);

    if(!loadDRCGains(robot_controller)) return CallbackReturn::ERROR;

    ee_names_.clear();
    for (const auto & name : params_.robot_name)
    {
        std::string ee_name = name + "_" + arm_id_ + "_";
        if (has_hand) ee_name = ee_name + "hand_tcp";
        else          ee_name = ee_name + "link8";
        ee_names_.push_back(ee_name);
    }


    model_updater_->setNode(get_node());
    model_updater_->setInterfaceFlags(has_position_state_interface_, has_velocity_state_interface_, has_effort_state_interface_,
                                      has_position_command_interface_, has_velocity_command_interface_, has_effort_command_interface_);
    model_updater_->initialize(num_robots_, manipulator_dof_, dt_, params_.robot_name, ee_names_);
    model_updater_->setDRCRobotData(std::move(robot_data));
    model_updater_->setDRCRobotController(std::move(robot_controller));

    
    // odom_timer_ = get_node()->create_wall_timer(
    //     std::chrono::duration<double>(1.0 / publish_rate_),
    //     [this]()
    //     {
    //         this->publishFromMobileStateBuffer();
    //     }
    // );


    heavy_init_done_ = true;
    LOGI(get_node(), "Heavy initialization done.");
    return CallbackReturn::SUCCESS;

}






CallbackReturn FR3HuskyActionController::on_activate(const rclcpp_lifecycle::State & /*previous_state*/)
{
    // if (initialize_heavy_resources() != CallbackReturn::SUCCESS)
    // {
    //     LOGE(get_node(), "Heavy initialization failed.");
    //     return CallbackReturn::ERROR;
    // }

    if (!model_updater_)
    {
        LOGE(get_node(), "Model updater is not configured.");
        return CallbackReturn::ERROR;
    }


    
    // Register manipulator interfaces into JointHandle
    std::vector<std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>>> state_by_type(allowed_interface_types_.size()); // [interface_type][joint_idx]
    std::vector<std::reference_wrapper<hardware_interface::LoanedCommandInterface>> cmd_by_type; // [joint_idx]

    for (const auto & interface : params_.manipulator_state_interfaces)
    {
        auto it = std::find(allowed_interface_types_.begin(), allowed_interface_types_.end(), interface);
        if (it == allowed_interface_types_.end()) continue;
        size_t idx = static_cast<size_t>(std::distance(allowed_interface_types_.begin(), it));
        if (!controller_interface::get_ordered_interfaces(state_interfaces_, params_.manipulator_joints, interface, state_by_type[idx]))
        {
            LOGE(get_node(), "Expected %zu '%s' state interfaces, got %zu.", manipulator_dof_, interface.c_str(), state_by_type[idx].size());
            return CallbackReturn::ERROR;
        }
        if (state_by_type[idx].size() != manipulator_dof_)
        {
            LOGE(get_node(), "State interface '%s' not available for every manipulator joint (%zu of %zu).", interface.c_str(), state_by_type[idx].size(), manipulator_dof_);
            return CallbackReturn::ERROR;
        }
    }


    const auto & cmd_interface = params_.manipulator_command_interface;
    auto it = std::find(allowed_interface_types_.begin(), allowed_interface_types_.end(), cmd_interface);
    if (it == allowed_interface_types_.end())
    {
        LOGE(get_node(), "Invalid command_interface '%s'.", cmd_interface.c_str());
        return CallbackReturn::ERROR;
    }
    if (!controller_interface::get_ordered_interfaces(command_interfaces_, params_.manipulator_joints, cmd_interface, cmd_by_type))
    {
        LOGE(get_node(), "Expected %zu '%s' command interfaces, got %zu.",
            manipulator_dof_, cmd_interface.c_str(), cmd_by_type.size());
        return CallbackReturn::ERROR;
    }
    if (cmd_by_type.size() != manipulator_dof_)
    {
        LOGE(get_node(), "Command interface '%s' not available for every manipulator joint (%zu of %zu).",
            cmd_interface.c_str(), cmd_by_type.size(), manipulator_dof_);
        return CallbackReturn::ERROR;
    }


    RobotHandle robot_handle;
    robot_handle.mani_joints.reserve(manipulator_dof_);
    for (size_t i = 0; i < manipulator_dof_; ++i)
    {
        robot_handle.mani_joints.emplace_back(cmd_by_type[i]);
        auto& handle = robot_handle.mani_joints.back();
        handle.state.clear();

        auto default_state_ref = state_by_type[kPositionIndex][i]; // position state is mandatory
        handle.state.assign(allowed_interface_types_.size(), default_state_ref);

        if (has_position_state_interface_)   handle.state[kPositionIndex] = state_by_type[kPositionIndex][i];
        if (has_velocity_state_interface_)   handle.state[kVelocityIndex] = state_by_type[kVelocityIndex][i];
        if (has_effort_state_interface_ && state_by_type.size() > kEffortIndex && !state_by_type[kEffortIndex].empty())
            handle.state[kEffortIndex] = state_by_type[kEffortIndex][i];
    }


    // Register wheel interfaces
    auto configure_side = [this](const std::string & side, const std::vector<std::string> & wheel_names, std::vector<WheelHandle> & registered_handles)
    {
        (void)side;
        registered_handles.clear();
        registered_handles.reserve(wheel_names.size());
        for (const auto & wheel_name : wheel_names)
        {
            const auto position_handle = std::find_if(
                state_interfaces_.cbegin(), state_interfaces_.cend(),
                [this, &wheel_name](const auto & iface)
                {
                    return iface.get_prefix_name() == wheel_name && iface.get_interface_name() == allowed_interface_types_[0];
                });
            if (position_handle == state_interfaces_.cend())
            {
                LOGE(get_node(), "Unable to obtain joint position state handle for %s", wheel_name.c_str());
                return false;
            }
            const auto velocity_handle = std::find_if(
                state_interfaces_.cbegin(), state_interfaces_.cend(),
                [this, &wheel_name](const auto & iface)
                {
                    return iface.get_prefix_name() == wheel_name && iface.get_interface_name() == allowed_interface_types_[1];
                });
            if (velocity_handle == state_interfaces_.cend())
            {
                LOGE(get_node(), "Unable to obtain joint velocity state handle for %s", wheel_name.c_str());
                return false;
            }
            const auto command_handle = std::find_if(
                command_interfaces_.begin(), command_interfaces_.end(),
                [this, &wheel_name](const auto & iface)
                {
                    return iface.get_prefix_name() == wheel_name && iface.get_interface_name() == allowed_interface_types_[1];
                });
            if (command_handle == command_interfaces_.end())
            {
                LOGE(get_node(), "Unable to obtain joint command handle for %s", wheel_name.c_str());
                return false;
            }
            std::vector<std::reference_wrapper<const hardware_interface::LoanedStateInterface>> state_handles;
            state_handles.reserve(2);
            state_handles.emplace_back(std::ref(*position_handle));
            state_handles.emplace_back(std::ref(*velocity_handle));
            registered_handles.emplace_back(WheelHandle{std::move(state_handles), std::ref(*command_handle)});
        }
        return true;
    };


    if (!configure_side("left", params_.left_wheel_names, robot_handle.left_wheels) ||
        !configure_side("right", params_.right_wheel_names, robot_handle.right_wheels))
    {
        return CallbackReturn::ERROR;
    }

    model_updater_->setRobotHandles(std::move(robot_handle));




    // Assign Franka semantic component state interfaces
    if (use_franka_model_)
    {
        try
        {
            for (const auto & model : franka_robot_model_)
            {
                model->assign_loaned_state_interfaces(state_interfaces_);
            }
            model_updater_->setFrankaModel(&franka_robot_model_);
        }
        catch (const std::exception & e)
        {
            LOGW(get_node(), "Franka semantic component state interfaces not available (%s).", e.what());
        }
    }



    const bool joy_connected = isJoyConnected();
    if (params_.use_estop && !joy_connected)
    {
        LOGE(get_node(), "use_estop=true but joystick is not connected on topic '%s'.", kJoyTopic);
        return CallbackReturn::ERROR;
    }
    if (!params_.use_estop && !joy_connected)
    {
        LOGW(get_node(), "Joystick is not connected on topic '%s'. e-stop is disabled and controller continues.", kJoyTopic);
    }



    model_updater_->has_hand_ = false;
    for (const auto & ee_name : ee_names_)
    {
        if (ee_name.find("hand") != std::string::npos)
        {
            model_updater_->has_hand_ = true;
            break;
        }
    }

    if (model_updater_->has_hand_)
    {
        for (const auto& name : params_.robot_name)
        {
            if (model_updater_->gripper_clients_.find(name) == model_updater_->gripper_clients_.end())
            {
                const std::string prefix = name + "_franka_gripper";
                GripperClients clients;
                clients.grasp  = rclcpp_action::create_client<franka_msgs::action::Grasp>(get_node(), prefix + "/grasp");
                clients.move   = rclcpp_action::create_client<franka_msgs::action::Move>(get_node(), prefix + "/move");
                clients.homing = rclcpp_action::create_client<franka_msgs::action::Homing>(get_node(), prefix + "/homing");
                model_updater_->gripper_clients_[name] = std::move(clients);
                LOGI(get_node(), "Gripper action clients created for '%s' (prefix: %s)", name.c_str(), prefix.c_str());
            }
        }
    }

    auto all_servers = servers::ActionServerManager::createAllFR3Husky(get_node(), *model_updater_);

    task_servers_.clear();
    controller_servers_.clear();

    for (auto& s : all_servers)
    {
        if (s->mode_ == fr3_husky_controller::servers::ActionServerManager::ServerMode::CONTROLLER)
            controller_servers_.push_back(s);
        else
            task_servers_.push_back(s);
    }

    active_task_.reset();

    if (!idle_control_)
    {
        idle_control_ = std::make_unique<servers::IdleControl>("fr3_husky_idle", get_node(), *model_updater_);
    }

    if (!odometry_publisher_)
    {
        odometry_publisher_ = get_node()->create_publisher<nav_msgs::msg::Odometry>("~/odom", rclcpp::SystemDefaultsQoS());
    }

    if (!odometry_transform_publisher_)
    {
        odometry_transform_publisher_ = get_node()->create_publisher<tf2_msgs::msg::TFMessage>("/tf", rclcpp::SystemDefaultsQoS());
    }

    mobi_state_pub_buf_.writeFromNonRT(std::make_pair(model_updater_->base_pose_w_, model_updater_->base_vel_b_));

    if (!odom_timer_)
    {
        odom_timer_ = get_node()->create_wall_timer(
            std::chrono::duration<double>(1.0 / publish_rate_),
            [this]()
            {
                this->publishFromMobileStateBuffer();
            }
        );
    }

    model_updater_->updateJointStates();
    model_updater_->updateRobotData();
    model_updater_->setInitFromCurrent();
    maybeRandomizeSceneSpawn(get_node(), scene_spawn_randomized_);
    model_updater_->updateJointStates();
    model_updater_->updateRobotData();
    model_updater_->setInitFromCurrent();

    is_halted_ = false;

    play_time_ = get_node()->now().seconds();
    control_start_time_ = play_time_;



    // model_updater_->updateJointStates();
    // model_updater_->updateRobotData();
    // model_updater_->setInitFromCurrent();


    return CallbackReturn::SUCCESS;
}




CallbackReturn FR3HuskyActionController::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/)
{
    if (!is_halted_)
    {
        if(model_updater_) model_updater_->haltCommands();
        is_halted_ = true;
    }

    odom_timer_.reset();
    estop_is_active_ = false;
    estop_button_pressed_.store(false, std::memory_order_release);
    joy_msg_received_.store(false, std::memory_order_release);
    scene_spawn_randomized_ = false;

    return CallbackReturn::SUCCESS;
}


controller_interface::return_type FR3HuskyActionController::update(const rclcpp::Time& time, const rclcpp::Duration& period)
{

    if (!model_updater_)
    {
        LOGE(get_node(), "Model updater is not available during update loop.");
        return controller_interface::return_type::ERROR;
    }

    if (get_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE)
    {
        if (!is_halted_)
        {
            model_updater_->haltCommands();
            is_halted_ = true;
        }
        return controller_interface::return_type::OK;
    }

    if (params_.use_estop && !isJoyConnected())
    {
        LOGE(get_node(), "Joystick disconnected while e-stop is enabled.");
        return controller_interface::return_type::ERROR;
    }

    play_time_ = get_node()->now().seconds();
    model_updater_->updateJointStates();
    model_updater_->updateRobotData();

    mobi_state_pub_buf_.writeFromNonRT(
        std::make_pair(model_updater_->base_pose_w_, model_updater_->base_vel_b_));

    // 1. active task 종료/취소 처리
    if (active_task_)
    {
        const bool cancel = active_task_->consumeCancelRequest();
        const bool still_active = active_task_->isActive();

        if (cancel || !still_active)
        {
            active_task_->onDeactivated();
            active_task_.reset();
        }
    }

    // 2. task activate 요청 스캔
    {
        std::shared_ptr<fr3_husky_controller::servers::ActionServerManager> best;
        int best_p = std::numeric_limits<int>::min();

        for (auto& s : task_servers_)
        {
            if (s->consumeActivateRequest())
            {
                const int p = s->priority();
                if (!best || p > best_p)
                {
                    best = s;
                    best_p = p;
                }
            }
        }

        if (best)
        {
            if (!active_task_)
            {
                active_task_ = best;
                active_task_->onActivated();
            }
            else if (best.get() != active_task_.get())
            {
                if (active_task_->canBePreempted() && best_p > active_task_->priority())
                {
                    RCLCPP_INFO(
                        get_node()->get_logger(),
                        "[Controller] Preempting [%s] (priority=%d) with [%s] (priority=%d)",
                        active_task_->getName().c_str(), active_task_->priority(),
                        best->getName().c_str(), best_p);

                    active_task_->onDeactivated();
                    active_task_.reset();

                    active_task_ = best;
                    active_task_->onActivated();
                }
                else
                {
                    RCLCPP_INFO(
                        get_node()->get_logger(),
                        "[Controller] Rejecting new task [%s] while [%s] is active",
                        best->getName().c_str(),
                        active_task_->getName().c_str());

                    best->onDeactivated();
                }
            }
        }
    }

    // 3. controller 서버는 항상 update
    for (auto& s : controller_servers_)
    {
        s->update(time, period);
    }

    bool any_controller_active = false;
    for (const auto& s : controller_servers_)
    {
        if (s->isActive())
        {
            any_controller_active = true;
            break;
        }
    }

    // 4. active task 실행 or idle
    if (active_task_)
    {
        if (idle_control_) idle_control_->onDeactivated();
        active_task_->update(time, period);
    }
    else if (any_controller_active)
    {
        if (idle_control_) idle_control_->onDeactivated();
    }
    else
    {
        if (idle_control_) idle_control_->compute(time, period);
    }


    const bool estop_pressed =
        params_.use_estop &&
        joy_msg_received_.load(std::memory_order_acquire) &&
        estop_button_pressed_.load(std::memory_order_acquire);

    if (estop_pressed && !estop_is_active_)
    {
        LOGW(get_node(), "E-STOP for mobile base activated!");
    }
    else if (!estop_pressed && estop_is_active_)
    {
        LOGI(get_node(), "E-STOP for mobile base released!");
    }
    estop_is_active_ = estop_pressed;

    if (estop_is_active_)
    {
        model_updater_->forceStopMobile();
    }

    return controller_interface::return_type::OK;
}



bool FR3HuskyActionController::setJointIndex(const std::string& urdf_xml, drc::MobileManipulator::JointIndex& out_idx)
{
    // Init output
    out_idx.virtual_start = -1;
    out_idx.mani_start    = -1;
    out_idx.mobi_start    = -1;

    pinocchio::Model model;
    pinocchio::urdf::buildModelFromXML(urdf_xml, model);

    const int nj = (int)model.njoints;

    auto join_keys = [](const std::vector<std::string>& keys) -> std::string
    {
        std::ostringstream os;
        for (size_t i = 0; i < keys.size(); ++i)
        {
            if (i) os << ", ";
            os << "'" << keys[i] << "'";
        }
        return os.str();
    };

    auto preview_names = [&](const std::vector<int>& idxs) -> std::string
    {
        std::ostringstream os;
        const size_t limit = 6;
        for (size_t i = 0; i < idxs.size() && i < limit; ++i)
        {
            if (i) os << ", ";
            os << model.names[idxs[i]];
        }
        if (idxs.size() > limit) os << ", ...";
        return os.str();
    };

    // ---------- Common search lambda with detailed diagnostics ----------
    auto find_block = [&](const std::string& label,
                          const std::vector<std::string>& keys,
                          int N,
                          int& out_start_vs,
                          bool ignore_dummy = false) -> bool
    {
        if (N <= 0)
        {
            LOGE(get_node(), "[setJointIndex:%s] requested block size N=%d is invalid.", label.c_str(), N);
            return false;
        }

        std::vector<int> matches;
        matches.reserve(nj);
        for (int i = 1; i < nj; ++i) // skip universe joint (0)
        {
            const std::string& name = model.names[i];
            if (ignore_dummy && name.find("dummy") != std::string::npos) continue;
            bool match = true;
            for (const auto& key : keys)
            {
                if (name.find(key) == std::string::npos)
                {
                    match = false;
                    break;
                }
            }
            if (match) matches.push_back(i);
        }

        if ((int)matches.size() < N)
        {
            LOGE(get_node(),
                 "[setJointIndex:%s] Expected %d joints but found only %zu joints containing keys (%s). Example matches: %s",
                 label.c_str(), N, matches.size(), join_keys(keys).c_str(), preview_names(matches).c_str());
            return false;
        }

        std::string last_reason;
        int attempts = 0;
        for (int start = 1; start + N - 1 < nj; ++start)
        {
            ++attempts;
            std::string reason;

            // 1) keyword check
            for (int k = 0; k < N; ++k)
            {
                const std::string& name = model.names[start + k];
                if (ignore_dummy && name.find("dummy") != std::string::npos)
                {
                    reason = "Joint '" + name + "' marked as dummy; skipping.";
                    break;
                }
                for (const auto& key : keys)
                {
                    if (name.find(key) == std::string::npos)
                    {
                        reason = "Joint '" + name + "' does not contain keyword '" + key + "'";
                        break;
                    }
                }
                if (!reason.empty()) break;
            }
            if (!reason.empty()) { last_reason = reason; continue; }

            // 2) 1-DoF continuity check (nvs/nqs)
            for (int k = 0; k < N; ++k)
            {
                if (model.nvs[start + k] != 1 || model.nqs[start + k] != 1)
                {
                    std::ostringstream os;
                    os << "Joint '" << model.names[start + k] << "' is not 1-DoF (nv=" << model.nvs[start + k]
                       << ", nq=" << model.nqs[start + k] << ")";
                    reason = os.str();
                    break;
                }
            }
            if (!reason.empty()) { last_reason = reason; continue; }

            // 3) velocity index contiguity check
            const int v0 = model.idx_vs[start];
            for (int k = 1; k < N; ++k)
            {
                if (model.idx_vs[start + k] != v0 + k)
                {
                    std::ostringstream os;
                    os << "Contiguous index mismatch: '" << model.names[start + k] << "' expected=" << v0 + k
                       << ", actual=" << model.idx_vs[start + k];
                    reason = os.str();
                    break;
                }
            }
            if (!reason.empty()) { last_reason = reason; continue; }

            out_start_vs = model.idx_vs[start];
            return true;
        }

        LOGE(get_node(),
             "[setJointIndex:%s] Found %zu joints containing keys (%s) but no contiguous block of %d. Last failure: %s (checked %d positions)",
             label.c_str(), matches.size(), join_keys(keys).c_str(), N,
             last_reason.c_str(), attempts);
        return false;
    };

    // 1) Virtual joints (v_*_joint) : 3
    int virtual_vs = -1;
    if (!find_block("virtual", { "v_", "_joint" }, 3, virtual_vs)) return false;

    // 2) FR3 manipulator joints
    const int mani_N = FR3_DOF * (int)(num_robots_);
    int mani_vs = -1;
    if (!find_block("fr3", { arm_id_, "joint" }, mani_N, mani_vs)) return false;

    // 3) Wheel joints
    int wheel_vs = -1;
    if (!find_block("mobile", { "wheel" }, 2, wheel_vs, true)) return false;

    out_idx.virtual_start = virtual_vs;
    out_idx.mani_start    = mani_vs;
    out_idx.mobi_start    = wheel_vs;

    return true;
}

void FR3HuskyActionController::publishFromMobileStateBuffer()
{
    const std::pair<Eigen::Affine2d, Eigen::Vector3d> s = *mobi_state_pub_buf_.readFromRT();
    const Eigen::Affine2d base_pose_w = s.first;
    const Eigen::Vector3d base_vel_b = s.second;

    // yaw from pose
    const double yaw = Eigen::Rotation2Dd(base_pose_w.linear()).angle();

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw);


    // Odometry publish 
    if (odometry_publisher_)
    {
        nav_msgs::msg::Odometry msg;
        msg.header.stamp = get_node()->now();   
        msg.header.frame_id = params_.odom_frame_id;
        msg.child_frame_id  = params_.base_frame_id;

        msg.pose.covariance.fill(0.0);
        msg.twist.covariance.fill(0.0);
        constexpr size_t N = 6;
        for (size_t i = 0; i < N; ++i)
        {
            const size_t diag = N * i + i;
            msg.pose.covariance[diag]  = params_.pose_covariance_diagonal[i];
            msg.twist.covariance[diag] = params_.twist_covariance_diagonal[i];
        }

        msg.pose.pose.position.x = base_pose_w.translation()(0);
        msg.pose.pose.position.y = base_pose_w.translation()(1);
        msg.pose.pose.orientation.x = q.x();
        msg.pose.pose.orientation.y = q.y();
        msg.pose.pose.orientation.z = q.z();
        msg.pose.pose.orientation.w = q.w();

        msg.twist.twist.linear.x  = base_vel_b(0);
        msg.twist.twist.angular.z = base_vel_b(2);

        odometry_publisher_->publish(msg);
    }

    // TF publish
    if (odometry_transform_publisher_)
    {
        tf2_msgs::msg::TFMessage tfm;
        tfm.transforms.resize(1);
        auto & t = tfm.transforms[0];

        t.header.stamp = get_node()->now();
        t.header.frame_id = params_.odom_frame_id;
        t.child_frame_id  = params_.base_frame_id;
        t.transform.translation.x = base_pose_w.translation()(0);
        t.transform.translation.y = base_pose_w.translation()(1);
        t.transform.rotation.x = q.x();
        t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z();
        t.transform.rotation.w = q.w();

        odometry_transform_publisher_->publish(tfm);
    }
}

void FR3HuskyActionController::onJoyMessage(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    joy_msg_received_.store(true, std::memory_order_release);

    if (!msg)
    {
        estop_button_pressed_.store(false, std::memory_order_release);
        return;
    }

    const int64_t button_index = params_.estop_button_index;
    if (button_index < 0 || static_cast<size_t>(button_index) >= msg->buttons.size())
    {
        if (!estop_button_index_warned_)
        {
            LOGW(get_node(),
                 "Invalid estop_button_index=%ld for Joy message (buttons size=%zu). e-stop is treated as released.",
                 static_cast<long>(button_index), msg->buttons.size());
            estop_button_index_warned_ = true;
        }
        estop_button_pressed_.store(false, std::memory_order_release);
        return;
    }

    estop_button_pressed_.store(msg->buttons[static_cast<size_t>(button_index)] != 0, std::memory_order_release);
}

bool FR3HuskyActionController::isJoyConnected() const
{
    return joy_subscriber_ && joy_subscriber_->get_publisher_count() > 0;
}

bool FR3HuskyActionController::loadDRCGains(std::shared_ptr<drc::MobileManipulator::RobotController> robot_controller)
{
    constexpr size_t task_dof = 6;
    constexpr size_t base_dof = 3;

    const auto & joint = params_.dyros_robot_controller.manipulator_joint_gains;
    const auto & task  = params_.dyros_robot_controller.task_gains;
    const auto & qpik  = params_.dyros_robot_controller.QPIK_weight;
    const auto & qpid  = params_.dyros_robot_controller.QPID_weight;

    auto check_vector_size = [this](const std::string & name, size_t expected, size_t got) -> bool
    {
        if (got != expected)
        {
            LOGE(get_node(), "Parameter '%s' expected %zu values, got %zu.", name.c_str(), expected, got);
            return false;
        }
        return true;
    };

    // manipulator joint gains
    if (!check_vector_size("dyros_robot_controller.manipulator_joint_gains.kp", manipulator_dof_, joint.kp.size())) return false;
    if (!check_vector_size("dyros_robot_controller.manipulator_joint_gains.kv", manipulator_dof_, joint.kv.size())) return false;

    // task-space gains
    if (!check_vector_size("dyros_robot_controller.task_gains.ik.kp", task_dof, task.ik.kp.size())) return false;
    if (!check_vector_size("dyros_robot_controller.task_gains.id.kp", task_dof, task.id.kp.size())) return false;
    if (!check_vector_size("dyros_robot_controller.task_gains.id.kv", task_dof, task.id.kv.size())) return false;

    // QPIK weights
    if (!check_vector_size("dyros_robot_controller.QPIK_weight.tracking.weights", task_dof, qpik.tracking.weights.size())) return false;
    if (!check_vector_size("dyros_robot_controller.QPIK_weight.joint.velocity.manipulator", manipulator_dof_, qpik.joint.velocity.manipulator.size())) return false;
    if (!check_vector_size("dyros_robot_controller.QPIK_weight.joint.acceleration.manipulator", manipulator_dof_, qpik.joint.acceleration.manipulator.size())) return false;
    if (!check_vector_size("dyros_robot_controller.QPIK_weight.joint.velocity.mobile", base_dof, qpik.joint.velocity.mobile.size())) return false;
    if (!check_vector_size("dyros_robot_controller.QPIK_weight.joint.acceleration.mobile", base_dof, qpik.joint.acceleration.mobile.size())) return false;

    // QPID weights
    if (!check_vector_size("dyros_robot_controller.QPID_weight.tracking.weights", task_dof, qpid.tracking.weights.size())) return false;
    if (!check_vector_size("dyros_robot_controller.QPID_weight.joint.velocity.manipulator", manipulator_dof_, qpid.joint.velocity.manipulator.size())) return false;
    if (!check_vector_size("dyros_robot_controller.QPID_weight.joint.acceleration.manipulator", manipulator_dof_, qpid.joint.acceleration.manipulator.size())) return false;
    if (!check_vector_size("dyros_robot_controller.QPID_weight.joint.velocity.mobile", base_dof, qpid.joint.velocity.mobile.size())) return false;
    if (!check_vector_size("dyros_robot_controller.QPID_weight.joint.acceleration.mobile", base_dof, qpid.joint.acceleration.mobile.size())) return false;

    const Eigen::VectorXd mani_joint_kp = Eigen::Map<const Eigen::VectorXd>(joint.kp.data(), joint.kp.size());
    const Eigen::VectorXd mani_joint_kv = Eigen::Map<const Eigen::VectorXd>(joint.kv.data(), joint.kv.size());

    const Eigen::VectorXd task_ik_kp = Eigen::Map<const Eigen::VectorXd>(task.ik.kp.data(), task.ik.kp.size());
    const Eigen::VectorXd task_id_kp = Eigen::Map<const Eigen::VectorXd>(task.id.kp.data(), task.id.kp.size());
    const Eigen::VectorXd task_id_kv = Eigen::Map<const Eigen::VectorXd>(task.id.kv.data(), task.id.kv.size());

    const Eigen::VectorXd qpik_tracking = Eigen::Map<const Eigen::VectorXd>(qpik.tracking.weights.data(), qpik.tracking.weights.size());
    const Eigen::VectorXd qpik_mani_damping = Eigen::Map<const Eigen::VectorXd>(qpik.joint.velocity.manipulator.data(), qpik.joint.velocity.manipulator.size());
    const Eigen::VectorXd qpik_mani_acc_damping = Eigen::Map<const Eigen::VectorXd>(qpik.joint.acceleration.manipulator.data(), qpik.joint.acceleration.manipulator.size());
    const Eigen::VectorXd qpik_mobi_damping = Eigen::Map<const Eigen::VectorXd>(qpik.joint.velocity.mobile.data(), qpik.joint.velocity.mobile.size());
    const Eigen::VectorXd qpik_mobi_acc_damping = Eigen::Map<const Eigen::VectorXd>(qpik.joint.acceleration.mobile.data(), qpik.joint.acceleration.mobile.size());
    
    const Eigen::VectorXd qpid_tracking = Eigen::Map<const Eigen::VectorXd>(qpid.tracking.weights.data(), qpid.tracking.weights.size());
    const Eigen::VectorXd qpid_mani_vel_damping = Eigen::Map<const Eigen::VectorXd>(qpid.joint.velocity.manipulator.data(), qpid.joint.velocity.manipulator.size());
    const Eigen::VectorXd qpid_mani_acc_damping = Eigen::Map<const Eigen::VectorXd>(qpid.joint.acceleration.manipulator.data(), qpid.joint.acceleration.manipulator.size());
    const Eigen::VectorXd qpid_mobi_vel_damping = Eigen::Map<const Eigen::VectorXd>(qpid.joint.velocity.mobile.data(), qpid.joint.velocity.mobile.size());
    const Eigen::VectorXd qpid_mobi_acc_damping = Eigen::Map<const Eigen::VectorXd>(qpid.joint.acceleration.mobile.data(), qpid.joint.acceleration.mobile.size());

    std::ostringstream oss;
    oss << "dyros robot controller gains" << "\n"
        << "\tmanipulator_joint_gains" << "\n"
        << "\t\tKp: " << mani_joint_kp.transpose() << "\n"
        << "\t\tKv: " << mani_joint_kv.transpose() << "\n"
        << "\ttask_gains" << "\n"
        << "\t\tik.Kp: " << task_ik_kp.transpose() << "\n"
        << "\t\tid.Kp: " << task_id_kp.transpose() << "\n"
        << "\t\tid.Kv: " << task_id_kv.transpose() << "\n"
        << "\tQPIK_weight" << "\n"
        << "\t\ttracking.weights: " << qpik_tracking.transpose() << "\n"
        << "\t\tjoint.velocity.manipulator: " << qpik_mani_damping.transpose() << "\n"
        << "\t\tjoint.acceleration.manipulator: " << qpik_mani_acc_damping.transpose() << "\n"
        << "\t\tjoint.velocity.mobile: " << qpik_mobi_damping.transpose() << "\n"
        << "\t\tjoint.acceleration.mobile: " << qpik_mobi_acc_damping.transpose() << "\n"
        << "\tQPID_weight" << "\n"
        << "\t\ttracking.weights: " << qpid_tracking.transpose() << "\n"
        << "\t\tjoint.velocity.manipulator: " << qpid_mani_vel_damping.transpose() << "\n"
        << "\t\tjoint.acceleration.manipulator: " << qpid_mani_acc_damping.transpose() << "\n"
        << "\t\tjoint.velocity.mobile: " << qpid_mobi_vel_damping.transpose() << "\n"
        << "\t\tjoint.acceleration.mobile: " << qpid_mobi_acc_damping.transpose();
    LOGI(get_node(), "%s", oss.str().c_str());

    robot_controller->setManipulatorJointGain(mani_joint_kp, mani_joint_kv);
    robot_controller->setIKGain(task_ik_kp);
    robot_controller->setIDGain(task_id_kp, task_id_kv);
    robot_controller->setQPIKGain(qpik_tracking, qpik_mani_damping, qpik_mani_acc_damping, qpik_mobi_damping, qpik_mobi_acc_damping);
    robot_controller->setQPIDGain(qpid_tracking, qpid_mani_vel_damping, qpid_mani_acc_damping, qpid_mobi_vel_damping, qpid_mobi_acc_damping);
    return true;
}
}  // namespace fr3_husky_controller

#include "class_loader/register_macro.hpp"
CLASS_LOADER_REGISTER_CLASS(fr3_husky_controller::FR3HuskyActionController, controller_interface::ControllerInterface);
