/**
Software License Agreement (BSD)

\file      create.h
\authors   Jacob Perron <jperron@sfu.ca>
\copyright Copyright (c) 2015, Autonomy Lab (Simon Fraser University), All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of Autonomy Lab nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CREATE_H
#define CREATE_H

#include <boost/shared_ptr.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <string>
#include <unistd.h>

#include "create/serial_stream.h"
#include "create/serial_query.h"
#include "create/data.h"
#include "create/types.h"
#include "create/util.h"

namespace create {

  enum SerialMode {
	AUTO = 0,
	STREAMING = 1,
	QUERY = 2
  };

  class Create {
    private:
      typedef boost::numeric::ublas::matrix<float> Matrix;

      enum CreateLED {
        LED_DEBRIS = 1,
        LED_SPOT = 2,
        LED_DOCK = 4,
        LED_CHECK = 8
      };


      RobotModel model;

      uint8_t mainMotorPower;
      uint8_t sideMotorPower;
      uint8_t vacuumMotorPower;

      // LEDs
      uint8_t debrisLED;
      uint8_t spotLED;
      uint8_t dockLED;
      uint8_t checkLED;
      uint8_t powerLED;
      uint8_t powerLEDIntensity;

      CreateMode mode;
      SerialMode serialMode;

      uint32_t prevTicksLeft;
      uint32_t prevTicksRight;
      float totalLeftDist;
      float totalRightDist;
      bool firstOnData;
      util::timestamp_t prevOnDataTime;

      void init(const SerialMode& serialMode = AUTO);
      // Add two matrices and handle overflow case
      Matrix addMatrices(const Matrix &A, const Matrix &B) const;
      void onData();
      bool updateLEDs();

    protected:
      boost::shared_ptr<create::Data> data;
      boost::shared_ptr<create::Serial> serial;

    public:
      /* Default constructor.
       * Does not attempt to establish serial connection to Create.
       */
      Create(RobotModel=RobotModel::CREATE_1, const SerialMode& serialMode = AUTO);

      /* Attempts to establish serial connection to Create.
       * \param port of your computer that is connected to Create.
       * \param baud rate to communicate with Create. Typically,
       *        115200 for Create 2 and 57600 for Create 1.
       * \param model type of robot.
       */
      Create(const std::string& port, const int& baud, RobotModel model = RobotModel::CREATE_1, const SerialMode& serialMode = AUTO);

      ~Create();

      /* Make a serial connection to Create.
       * This is the first thing that should be done after instantiated this class.
       * \return true if a successful connection is established, false otherwise.
       */
      bool connect(const std::string& port, const int& baud);

      inline bool connected() const { return serial->connected(); };

      /* Disconnect from serial.
       */
      void disconnect();

      /* Change Create mode.
       * \param mode to put Create in.
       * \return true if successful, false otherwise
       */
      bool setMode(const create::CreateMode& mode);

      /* Starts a cleaning mode.
       * Changes mode to MODE_PASSIVE.
       * \return true if successful, false otherwise
       */
      bool clean(const create::CleanMode& mode = CLEAN_DEFAULT);

      /* Starts the docking behaviour.
       * Changes mode to MODE_PASSIVE.
       * \return true if successful, false otherwise
       */
      bool dock() const;

      /* Sets the internal clock of Create.
       * \param day in range [0, 6]
       * \param hour in range [0, 23]
       * \param min in range [0, 59]
       * \return true if successful, false otherwise
       */
      bool setDate(const create::DayOfWeek& day, const uint8_t& hour, const uint8_t& min) const;

      /* Set the average wheel velocity and turning radius of Create.
       * \param velocity is in m/s bounded between [-0.5, 0.5]
       * \param radius in meters.
       *        Special cases: drive straight = CREATE_2_STRAIGHT_RADIUS,
       *                       turn in place counter-clockwise = CREATE_2_IN_PLACE_RADIUS,
       *                       turn in place clockwise = -CREATE_2_IN_PLACE_RADIUS
       * \return true if successful, false otherwise
       */
      bool driveRadius(const float& velocity, const float& radius);

      /* Set the velocities for the left and right wheels.
       * \param leftWheel velocity in m/s.
       * \param rightWheel veloctiy in m/s.
       * \return true if successful, false otherwise
       */
      bool driveWheels(const float& leftWheel, const float& rightWheel);

      /* Set the forward and angular velocity of Create.
       * \param xVel in m/s
       * \param angularVel in rads/s
       * \return true if successful, false otherwise
       */
      bool drive(const float& xVel, const float& angularVel);

      /* Set the power to the side brush motor.
       * \param power is in the range [-1, 1]
       * \return true if successful, false otherwise
       */
      bool setSideMotor(const float& power);

      /* Set the power to the main brush motor.
       * \param power is in the range [-1, 1]
       * \return true if successful, false otherwise
       */
      bool setMainMotor(const float& power);

      /* Set the power to the vacuum motor.
       * \param power is in the range [0, 1]
       * \return true if successful, false otherwise
       */
      bool setVacuumMotor(const float& power);

      /* Set the power of all motors.
       * \param mainPower in the range [-1, 1]
       * \param sidePower in the range [-1, 1]
       * \param vacuumPower in the range [0, 1]
       * \return true if successful, false otherwise
       */
      bool setAllMotors(const float& mainPower, const float& sidePower, const float& vacuumPower);

      /* Set the blue "debris" LED on/off.
       * \param enable
       * \return true if successful, false otherwise
       */
      bool enableDebrisLED(const bool& enable);

      /* Set the green "spot" LED on/off.
       * \param enable
       * \return true if successful, false otherwise
       */
      bool enableSpotLED(const bool& enable);

      /* Set the green "dock" LED on/off.
       * \param enable
       * \return true if successful, false otherwise
       */
      bool enableDockLED(const bool& enable);

      /* Set the orange "check Create" LED on/off.
       * \param enable
       * \return true if successful, false otherwise
       */
      bool enableCheckRobotLED(const bool& enable);

      /* Set the center power LED.
       * \param power in range [0, 255] where 0 = green and 255 = red
       * \param intensity in range [0, 255]
       * \return true if successful, false otherwise
       */
      bool setPowerLED(const uint8_t& power, const uint8_t& intensity = 255);

      /* Set the four 7-segment display digits from left to right.
       * \param segments to enable (true) or disable (false).
       *        The size of segments should be less than 29.
       *        The ordering of segments is left to right, top to bottom for each digit:
       *
       *            0           7             14            21
       *          |‾‾‾|       |‾‾‾|         |‾‾‾|         |‾‾‾|
       *        1 |___| 2   8 |___| 9    15 |___| 16   22 |___| 23
       *          | 3 |       | 10|         | 17|         | 24|
       *        4 |___| 5   11|___| 12   18 |___| 19   25 |___| 26
       *            6           13            20            27
       *
       * \return true if successful, false otherwise
       */
      //TODO (https://github.com/AutonomyLab/libcreate/issues/7)
      //bool setDigits(const std::vector<bool>& segments) const;

      /* Set the four 7-segment display digits from left to right with ASCII codes.
       * Any code out side the accepted ascii ranges results in blank display.
       * \param digit1 is left most digit with ascii range [32, 126]
       * \param digit2 is second to left digit with ascii range [32, 126]
       * \param digit3 is second to right digit with ascii range [32, 126]
       * \param digit4 is right most digit with ascii range [32, 126]
       * \return true if successful, false otherwise
       */
      bool setDigitsASCII(const uint8_t& digit1, const uint8_t& digit2,
                          const uint8_t& digit3, const uint8_t& digit4) const;

      /* Defines a song from the provided notes and labels it with a song number.
       * \param songNumber can be one of four possible song slots, [0, 4]
       * \param songLength is the number of notes, maximum 16.
       *        length(notes) = length(durations) = songLength should be true.
       * \param notes is a sequence of notes. Each note is in the range [31, 127].
       *        Anything outside this range is considered a rest note.
       * \param durations for each note in fractions of a second from the range [0, 4)
       * \return true if successful, false otherwise
       */
      bool defineSong(const uint8_t& songNumber,
                      const uint8_t& songLength,
                      const uint8_t* notes,
                      const float* durations) const;

      /* Play a previously created song.
       * This command will not work if a song was not already defined with the specified song number.
       * \param songNumber is one of four stored songs in the range [0, 4]
       * \return true if successful, false otherwise
       */
      bool playSong(const uint8_t& songNumber) const;

      /* True if a left or right wheeldrop is detected.
       */
      bool isWheeldrop() const;

      /* True if a left wheeldrop is detected.
       */
      bool isLeftWheeldrop() const;

      /* True if a right wheeldrop is detected.
       */
      bool isRightWheeldrop() const;

      /* True if a caster wheeldrop is detected.
       */
      bool isCasterWheeldrop() const;

      /* Returns true if left bumper is pressed, false otherwise.
       */
      bool isLeftBumper() const;

      /* Returns true if right bumper is pressed, false otherwise.
       */
      bool isRightBumper() const;

      /* True if wall is seen to right of Create, false otherwise.
       */
      bool isWall() const;

      /* True if there are any cliff detections, false otherwise.
       */
      bool isCliff() const;

      /* True if cliff is seen to left of Create, false otherwise.
       */
      bool isLeftCliff() const;

      /* True if cliff is seen to right of Create, false otherwise.
       */
      bool isRightCliff() const;

      /* True if cliff is seen to front-left of Create, false otherwise.
       */
      bool isFrontLeftCliff() const;

      /* True if cliff is seen to front-right of Create, false otherwise.
       */
      bool isFrontRightCliff() const;

      /* True if there is a virtual wall signal is being received.
       */
      bool isVirtualWall() const;

      //TODO (https://github.com/AutonomyLab/libcreate/issues/8)
      //bool isWheelOvercurrent() const;

      //TODO (https://github.com/AutonomyLab/libcreate/issues/8)
      //bool isMainBrushOvercurrent() const;

      //TODO (https://github.com/AutonomyLab/libcreate/issues/8)
      //bool isSideBrushOvercurrent() const;

      /* Get level of the dirt detect sensor.
       * \return value in range [0, 255]
       */
      uint8_t getDirtDetect() const;

      /* Get value of 8-bit IR character currently being received by omnidirectional sensor.
       * \return value in range [0, 255]
       */
      uint8_t getIROmni() const;

      /* Get value of 8-bit IR character currently being received by left sensor.
       * \return value in range [0, 255]
       */

      uint8_t getIRLeft() const;

      /* Get value of 8-bit IR character currently being received by right sensor.
       * \return value in range [0, 255]
       */
      uint8_t getIRRight() const;

      /* Get state of 'clean' button ('play' button on Create 1).
       */
      bool isCleanButtonPressed() const;

      /* Not supported by any firmware!
       */
      bool isClockButtonPressed() const;

      /* Not supported by any firmware!
       */
      bool isScheduleButtonPressed() const;

      /* Get state of 'day' button.
       */
      bool isDayButtonPressed() const;

      /* Get state of 'hour' button.
       */
      bool isHourButtonPressed() const;

      /* Get state of 'min' button.
       */
      bool isMinButtonPressed() const;

      /* Get state of 'dock' button ('advance' button on Create 1).
       */
      bool isDockButtonPressed() const;

      /* Get state of 'spot' button.
       */
      bool isSpotButtonPressed() const;

      /* Get battery voltage.
       * \return value in volts
       */
      float getVoltage() const;

      /* Get current flowing in/out of battery.
       * A positive current implies Create is charging.
       * \return value in amps
       */
      float getCurrent() const;

      /* Get the temperature of battery.
       * \return value in Celsius
       */
      int8_t getTemperature() const;

      /* Get battery's remaining charge.
       * \return value in amp-hours
       */
      float getBatteryCharge() const;

      /* Get estimated battery charge capacity.
       * \return in amp-hours
       */
      float getBatteryCapacity() const;

      /* Return true if farthest left light sensor detects an obstacle, false otherwise.
       */
      bool isLightBumperLeft() const;

      /* Return true if front left light sensor detects an obstacle, false otherwise.
       */
      bool isLightBumperFrontLeft() const;

      /* Return true if center left light sensor detects an obstacle, false otherwise.
       */
      bool isLightBumperCenterLeft() const;

      /* Return true if farthest right light sensor detects an obstacle, false otherwise.
       */
      bool isLightBumperRight() const;

      /* Return true if front right light sensor detects an obstacle, false otherwise.
       */
      bool isLightBumperFrontRight() const;

      /* Return true if center right light sensor detects an obstacle, false otherwise.
       */
      bool isLightBumperCenterRight() const;

      /* Return the signal strength from the left light sensor.
       * \return value in range [0, 4095]
       */
      uint16_t getLightSignalLeft() const;

      /* Return the signal strength from the front-left light sensor.
       * \return value in range [0, 4095]
       */
      uint16_t getLightSignalFrontLeft() const;

      /* Return the signal strength from the center-left light sensor.
       * \return value in range [0, 4095]
       */
      uint16_t getLightSignalCenterLeft() const;

      /* Return the signal strength from the right light sensor.
       * \return value in range [0, 4095]
       */
      uint16_t getLightSignalRight() const;

      /* Return the signal strength from the front-right light sensor.
       * \return value in range [0, 4095]
       */
      uint16_t getLightSignalFrontRight() const;

      /* Return the signal strength from the center-right light sensor.
       * \return value in range [0, 4095]
       */
      uint16_t getLightSignalCenterRight() const;

      /* Return the signal strength from the wall sensor.
	   * \return value in range [0, 4095]
	   */
      uint16_t getWallSignal() const;

      /* Return the signal strength from the left cliff sensor.
	   * \return value in range [0, 4095]
	   */
      uint16_t getCliffLeftSignal() const;

      /* Return the signal strength from the front-left cliff sensor.
	   * \return value in range [0, 4095]
	   */
      uint16_t getCliffFrontLeftSignal() const;

      /* Return the signal strength from the front-right cliff sensor.
	   * \return value in range [0, 4095]
	   */
      uint16_t getCliffFrontRightSignal() const;

      /* Return the signal strength from the right cliff sensor.
	   * \return value in range [0, 4095]
	   */
      uint16_t getCliffRightSignal() const;

      /* Return true if Create is moving forward, false otherwise.
       */
      bool isMovingForward() const;

      /* Get the total distance (in meters) the left wheel has moved.
       */
      float getLeftWheelDistance() const;

      /* Get the total distance (in meters) the right wheel has moved.
       */
      float getRightWheelDistance() const;

      /* Get the requested velocity (in mm/sec) of the left wheel.
       * This value is bounded at the maximum velocity of the robot model.
       */
      int16_t getRequestedLeftWheelVel() const;

      /* Get the requested velocity (in mm/sec) of the right wheel.
       * This value is bounded at the maximum velocity of the robot model.
       */
      int16_t getRequestedRightWheelVel() const;

      /* Get the current charging state.
       */
      create::ChargingState getChargingState() const;

      /* Get the current mode reported by Create.
       */
      create::CreateMode getMode();

      /* Get the number of corrupt serial packets since first connecting to Create.
       */
      uint64_t getNumCorruptPackets() const;

      /* Get the total number of serial packets (including corrupt packets) since first connecting to Create.
       */
      uint64_t getTotalPackets() const;
  };  // end Create class

}  // namespace create

#endif  // CREATE_DRIVER_H
