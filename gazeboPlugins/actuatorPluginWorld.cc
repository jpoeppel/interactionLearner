#include <boost/bind.hpp>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/gazebo.hh>
#include "actuatorCommand.pb.h"
#include "modelState_v.pb.h"
#include "modelState.pb.h"
#include "worldState.pb.h"
#include <gazebo/sensors/sensors.hh>

#include <iostream>
#include <stdio.h>

namespace gazebo
{
  class ActuatorPlugin : public WorldPlugin
  {
    /* Custom world plugin for the simulation in order to publish the required 
       information for each object as well as react to the acutator commands. */

    // Pointer to the actuator
    private: physics::ModelPtr actuator;
    //Pointer to the world
    private: physics::WorldPtr world;
    // Pointer to the target
    private: physics::ModelPtr target;

    private: gazeboPlugins::msgs::ActuatorCommand::Command curCmd;
    private: math::Vector3 curDir;

    private: sensors::ContactSensorPtr contactSensor;
    // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;
    private: event::ConnectionPtr contactConnection;
    private: transport::SubscriberPtr msgSubscriber;
    private: transport::SubscriberPtr predictionSubscriber;
    private: transport::SubscriberPtr sensorSubscriber;
    private: transport::PublisherPtr worldStatePub;

    private: bool hasTarget;
    private: bool targetReached;

    typedef const boost::shared_ptr<const gazeboPlugins::msgs::ActuatorCommand> CmdPtr;
    void cb(CmdPtr &_msg)
    {
      this->curCmd = _msg->cmd();
      this->curDir = math::Vector3(_msg->direction().x(),_msg->direction().y(),_msg->direction().z());
    }

    typedef const boost::shared_ptr<const gazeboPlugins::msgs::ModelState_V> ModelSVPtr;
    void poseCB(ModelSVPtr &_msg)
    {

        for (unsigned int i=0; i<_msg->models_size();i++) {
            physics::ModelPtr m = this->world->GetModel(_msg->models(i).name());
            m->SetWorldPose(msgs::Convert(_msg->models(i).pose()));
        }

    }

    typedef const boost::shared_ptr<const gazebo::msgs::Sensor> SensorPtr;
    void sensorCB(SensorPtr &_msg)
    {
      std::cout << "Setting update rate to: " << _msg->update_rate() << std::endl;
      this->contactSensor->SetUpdateRate(_msg->update_rate());
    }

    public: void Load(physics::WorldPtr _parent, sdf::ElementPtr /*_sdf*/)
    {

      // Store the pointer to the model
      this->world = _parent;
      this->actuator = this->world->GetModel("actuator");
      this->contactSensor = boost::dynamic_pointer_cast<sensors::ContactSensor>(sensors::get_sensor("actuatorContact"));
      this->hasTarget = false;
      // Create our node for communication
      gazebo::transport::NodePtr node(new gazebo::transport::Node());
      node->Init("default");

      // Listen to custom topic
      this->msgSubscriber = node->Subscribe("/gazebo/default/actuatorMsg", &ActuatorPlugin::cb, this);
      this->predictionSubscriber = node->Subscribe("/gazebo/default/poses", &ActuatorPlugin::poseCB, this);
      this->sensorSubscriber = node->Subscribe("~/sensor", &ActuatorPlugin::sensorCB, this);

      this->worldStatePub = node->Advertise<gazeboPlugins::msgs::WorldState>("~/worldstate");
      // Listen to the update event. This event is broadcast every
      // simulation iteration.
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          boost::bind(&ActuatorPlugin::OnUpdate, this, _1));
      this->contactConnection = this->contactSensor->ConnectUpdated(
      boost::bind(&ActuatorPlugin::OnContact, this));
      // Make sure the parent sensor is active.
      this->contactSensor->SetActive(true);
      std::cout << "loaded plugin" << std::endl;
    }



    // Called by the world update start event
    public: void OnUpdate(const common::UpdateInfo & /*_info*/)
    {
      // Apply a small linear velocity to the model.
      if (not this->isActuatorMovementOk()) {
        std::cout << "Actuator OFB" << std::endl;
        this->curDir = math::Vector3(0.0,0.0,0.0);
        this->actuator->SetLinearVel(math::Vector3(0.0,0.0,0.0));
      }
      this->actuator->SetAngularVel(math::Vector3(0.0,0.0,0.0));
      if (this->curCmd == gazeboPlugins::msgs::ActuatorCommand::MOVE)
      {
        this->actuator->SetLinearVel(this->curDir);
        math::Pose p = this->actuator->GetWorldPose();
        this->actuator->SetWorldPose(math::Pose(math::Vector3(p.pos.x,p.pos.y,0.03), p.rot));
      }
    }

    public: void OnContact()
    {

      // Publish world state
      gazeboPlugins::msgs::WorldState worldState;
      gazeboPlugins::msgs::ModelState_V* models = worldState.mutable_model_v();
      physics::Model_V allModels = this->world->GetModels();
      for (unsigned int i = 0; i<allModels.size();i++)
      {
        physics::ModelPtr m = allModels[i];
        gazeboPlugins::msgs::ModelState* tmp = models->add_models();
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


        //Publish contacts
        msgs::Contacts* contacts = worldState.mutable_contacts();

        msgs::Contacts tmpC =this->contactSensor->GetContacts();
        contacts->CopyFrom(tmpC);
        msgs::Set(contacts->mutable_time(), this->world->GetSimTime());

        this->worldStatePub->Publish(worldState);
    }


    private: bool isActuatorMovementOk()
    {
        return ((this->actuator->GetWorldPose().pos+this->curDir).Distance(0.0,0.0,1.0) <= 2.5);
    }



  };

  // Register this plugin with the simulator
  GZ_REGISTER_WORLD_PLUGIN(ActuatorPlugin)
}
