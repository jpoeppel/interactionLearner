#include <iostream>
#include <sdf/sdf.hh>

#include "gazebo/gazebo.hh"
#include "gazebo/common/common.hh"
#include "gazebo/transport/transport.hh"
#include "gazebo/msgs/msgs.hh"


using namespace std;

int main(int argc, char * argv[])
{

    gazebo::msgs::GzString request;
    request.set_data("Hi");


    gazebo::transport::init();
    gazebo::transport::run();
    gazebo::transport::NodePtr node(new gazebo::transport::Node());
    node->Init("default");

    gazebo::transport::PublisherPtr imagePub =
            node->Advertise<gazebo::msgs::GzString>("/gazebo/default/gripperMsg");
    imagePub->WaitForConnection();
    printf("sending\n");
    imagePub->Publish(request);
    printf("was send\n");
    gazebo::transport::fini();
    return 0;

}
