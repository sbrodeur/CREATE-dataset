/******************************************************************************
 * 
 * Copyright (c) 2014, Simon Brodeur
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 *  - Redistributions of source code must retain the above copyright notice, 
 *    this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright notice, 
 *    this list of conditions and the following disclaimer in the documentation 
 *    and/or other materials provided with the distribution.
 *  - Neither the name of the NECOTIS research group nor the names of its contributors 
 *    may be used to endorse or promote products derived from this software 
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/FluidPressure.h>
#include <boost/lexical_cast.hpp>

using namespace std;


class CaptureNode {

    public:
        ros::NodeHandle node_;
        ros::Publisher pubPres_;
        std::string output_;
        double rate_;

        std::ifstream mPresFile_;
        sensor_msgs::FluidPressure msgPres_;
        char currentPressure_[sizeof(double)];

        CaptureNode() :
            node_("~") {

                std::string device;
                node_.param("device", device, std::string("/sys/bus/i2c/drivers/bmp085/2-0077/pressure0_input"));
                node_.param("rate", rate_, 0.0);
                node_.param("output", output_, std::string("imu/baro"));

                msgPres_.header.frame_id = "imu_link";

                pubPres_ = node_.advertise<sensor_msgs::FluidPressure>(output_, 20);

                mPresFile_.open(device.c_str(), std::ifstream::in);
            }

        virtual ~CaptureNode() {
            // Close files
            mPresFile_.close();
        }

        bool spin() {

        	ros::Rate rate(0.0);
        	if (rate_ > 0.0){
            	rate = ros::Rate(rate_);
        	}
            while (node_.ok()) {

                if(mPresFile_.good())
                {
                	mPresFile_.seekg(0, mPresFile_.beg);
                	mPresFile_.getline((char*) currentPressure_, sizeof(currentPressure_));
                }
                else
                {
                	ROS_WARN_THROTTLE(1, "Could not access Pressure file!");
                }

                //Publish Pressure
                msgPres_.header.stamp = ros::Time::now();
                msgPres_.fluid_pressure = float(boost::lexical_cast<double>(currentPressure_));
                pubPres_.publish(msgPres_);

                if (rate_ > 0.0){
                	rate.sleep();
                }
            }
            
            return true;
        }

};

int main(int argc, char **argv) {
    ros::init(argc, argv, "capture_baro");

    CaptureNode a;
    a.spin();
    return 0;
}
