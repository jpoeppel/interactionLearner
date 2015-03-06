#include <boost/bind.hpp>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/gazebo.hh>
#include "gripperCommand.pb.h"

#include <iostream>
#include <stdio.h>

namespace gazebo
{
  class GripperPlugin : public WorldPlugin
  {

    // Pointer to the gripper
    private: physics::ModelPtr gripper;
    //Pointer to the world
    private: physics::WorldPtr world;
    // Pointer to the gripper
    private: physics::ModelPtr target;

    // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;
    private: transport::SubscriberPtr msgSubscriber;

    private: bool hasTarget;
    private: bool targetReached;

    typedef const boost::shared_ptr<const gazeboPlugins::msgs::GripperCommand> CmdPtr;
    void cb(CmdPtr &_msg)
    {
      // Dump the message contents to stdout.
      std::cout << _msg->cmd();
      std::cout << "Recieving \n";
      std::cout << math::Vector3(_msg->direction().x(),_msg->direction().y(),_msg->direction().z())  ;// << _msg->direction().y << ", " << _msg->direction().z;
    }

    public: void Load(physics::WorldPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      std::cout << "loading plugin" << std::endl;
      // Store the pointer to the model
      this->world = _parent;
      this->gripper = this->world->GetModel("gripper");
      this->hasTarget = false;
      // Create our node for communication
      gazebo::transport::NodePtr node(new gazebo::transport::Node());
      node->Init("default");

      // Listen to custom topic
      msgSubscriber = node->Subscribe("/gazebo/default/gripperMsg", &GripperPlugin::cb, this);
      // Listen to the update event. This event is broadcast every
      // simulation iteration.
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          boost::bind(&GripperPlugin::OnUpdate, this, _1));
    }



    // Called by the world update start event
    public: void OnUpdate(const common::UpdateInfo & /*_info*/)
    {
      // Apply a small linear velocity to the model.
      if (this->hasTarget and not this->targetReached){
        math::Pose targetPose = this->target->GetWorldPose();
        targetPose.rot = this->gripper->GetWorldPose().rot;
        this->gripper->GetChildLink("finger")->SetWorldPose(targetPose);
      }


    }


  };

  // Register this plugin with the simulator
  GZ_REGISTER_WORLD_PLUGIN(GripperPlugin)
}
