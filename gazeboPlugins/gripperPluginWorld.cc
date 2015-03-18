#include <boost/bind.hpp>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/gazebo.hh>
#include "gripperCommand.pb.h"
#include "modelState_v.pb.h"
#include "modelState.pb.h"
#include <gazebo/sensors/sensors.hh>

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
    // Pointer to the target
    private: physics::ModelPtr target;

    private: gazeboPlugins::msgs::GripperCommand::Command curCmd;
    private: math::Vector3 curDir;

    private: sensors::ContactSensorPtr contactSensor;
    // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;
    private: event::ConnectionPtr contactConnection;
    private: transport::SubscriberPtr msgSubscriber;
    private: transport::PublisherPtr worldStatePub;

    private: bool hasTarget;
    private: bool targetReached;

    typedef const boost::shared_ptr<const gazeboPlugins::msgs::GripperCommand> CmdPtr;
    void cb(CmdPtr &_msg)
    {
      // Dump the message contents to stdout.
      std::cout << gazeboPlugins::msgs::GripperCommand::Command_Name(_msg->cmd()) << std::endl;
      curCmd = _msg->cmd();
      std::cout << "Recieving \n";
      curDir = math::Vector3(_msg->direction().x(),_msg->direction().y(),_msg->direction().z());
      std::cout << curDir << std::endl; // << _msg->direction().y << ", " << _msg->direction().z;

    }

    public: void Load(physics::WorldPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      std::cout << "loading plugin" << std::endl;
      // Store the pointer to the model
      this->world = _parent;
      this->gripper = this->world->GetModel("gripper");
      this->contactSensor = boost::dynamic_pointer_cast<sensors::ContactSensor>(sensors::get_sensor("gripperContact"));
      this->hasTarget = false;
      // Create our node for communication
      gazebo::transport::NodePtr node(new gazebo::transport::Node());
      node->Init("default");

      // Listen to custom topic
      this->msgSubscriber = node->Subscribe("/gazebo/default/gripperMsg", &GripperPlugin::cb, this);


      this->worldStatePub = node->Advertise<gazeboPlugins::msgs::ModelState_V>("~/worldstate");
      this->worldStatePub->WaitForConnection();
      // Listen to the update event. This event is broadcast every
      // simulation iteration.
 //     this->updateConnection = event::Events::ConnectWorldUpdateBegin(
 //         boost::bind(&GripperPlugin::OnUpdate, this, _1));
      this->contactConnection = this->contactSensor->ConnectUpdated(
      boost::bind(&GripperPlugin::OnContact, this));
      // Make sure the parent sensor is active.
      this->contactSensor->SetActive(true);
    }



    // Called by the world update start event
    public: void OnUpdate(const common::UpdateInfo & /*_info*/)
    {
      // Publish world state
      gazeboPlugins::msgs::ModelState_V models;
      physics::Model_V allModels = this->world->GetModels();
      for (unsigned int i = 0; i<allModels.size();i++)
      {
        physics::ModelPtr m = allModels[i];
        gazeboPlugins::msgs::ModelState* tmp = models.add_models();
        tmp->set_name(m->GetName());
        tmp->set_id(m->GetId());
        tmp->set_is_static(m->IsStatic());
        msgs::Pose* p = tmp->mutable_pose();
        msgs::Set(p, m->GetWorldPose());
        msgs::Vector3d* linvel = tmp->mutable_linvel();
        msgs::Set(linvel, m->GetWorldLinearVel());
        msgs::Vector3d* angvel = tmp->mutable_angvel();
        msgs::Set(angvel, m->GetWorldAngularVel());
      }

      this->worldStatePub->WaitForConnection();
      this->worldStatePub->Publish(models);

    }

    public: void OnContact()
    {

        // Apply a small linear velocity to the model.
      if (not this->isGripperMovementOk()) {
        std::cout << "Gripper OFB" << std::endl;
        this->curDir = math::Vector3(0.0,0.0,0.0);
      }
      this->gripper->SetLinearVel(this->curDir);
      this->gripper->SetAngularVel(math::Vector3(0.0,0.0,0.0));

      // Publish world state
      gazeboPlugins::msgs::ModelState_V models;
      physics::Model_V allModels = this->world->GetModels();
      for (unsigned int i = 0; i<allModels.size();i++)
      {
        physics::ModelPtr m = allModels[i];
        gazeboPlugins::msgs::ModelState* tmp = models.add_models();
        tmp->set_name(m->GetName());
        tmp->set_id(m->GetId());
        tmp->set_is_static(m->IsStatic());
        msgs::Pose* p = tmp->mutable_pose();
        msgs::Set(p, m->GetWorldPose());
        msgs::Vector3d* linvel = tmp->mutable_linvel();
        msgs::Set(linvel, m->GetWorldLinearVel());
        msgs::Vector3d* angvel = tmp->mutable_angvel();
        msgs::Set(angvel, m->GetWorldAngularVel());
      }

      this->worldStatePub->WaitForConnection();
      this->worldStatePub->Publish(models);

        //Publish contacts
        msgs::Contacts contacts;
        contacts = this->contactSensor->GetContacts();
        if (contacts.contact_size() > 0)
        {
            std::cout << "onContact" << std::endl;
        }
        for (unsigned int i = 0; i < contacts.contact_size(); ++i)
          {
            std::cout << "Collision between[" << contacts.contact(i).collision1()
                      << "] and [" << contacts.contact(i).collision2() << "]\n";

            for (unsigned int j = 0; j < contacts.contact(i).position_size(); ++j)
            {
              std::cout << j << "  Position:"
                        << contacts.contact(i).position(j).x() << " "
                        << contacts.contact(i).position(j).y() << " "
                        << contacts.contact(i).position(j).z() << "\n";
              std::cout << "   Normal:"
                        << contacts.contact(i).normal(j).x() << " "
                        << contacts.contact(i).normal(j).y() << " "
                        << contacts.contact(i).normal(j).z() << "\n";
              std::cout << "   Depth:" << contacts.contact(i).depth(j) << "\n";
            }
          }

    }


    private: bool isGripperMovementOk()
    {
        return ((this->gripper->GetWorldPose().pos+this->curDir).Distance(0.0,0.0,1.0) <= 2.5);
    }



  };

  // Register this plugin with the simulator
  GZ_REGISTER_WORLD_PLUGIN(GripperPlugin)
}
