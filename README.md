# Memory-based online learning of simple object manipulation

Repository I used to manage all files related to my master thesis. Included are the source codes of for the actual project in the python folder. The required gazebo worlds and plugins, the actual thesis as well as the slides for the presentation can be found in their respective folders. 


## Requirements

The actual learner only requires *numpy* as external dependency. In order to be able to talk to the simulation, pygazebo (https://github.com/jpieper/pygazebo) is required as well.

As simulation I have used Gazebo 2.2.3, but newer versions should work as well with at most little modifications.

## Install

Assuming numpy and pygazebo have already been installed, only the gazeboPlugin as well as the used protoBuf files need to be compiled using the provided cmake file.
Compiling the plugin requires the plugin to be found. Currently I use an environment variable masterPath that points
to the parent directory of this folder.

To be able to build the plugins:
```
export masterPath=[path_to_master]
```

## Run

In order to run the program as is, a gazebo server needs to be started with a suitable worldfile which uses the provided plugin to talk to the Python program.

To be able to load the plugins, their path needs to be exposed to gazebo:
```
export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:[path_to_masterThesis]/gazeboPlugins/build
```

Afterwards the server can be started using:

```
gzserver [path_to_Worldfile]
```

Once the server is running (which can be checked using *gzclient*), the gazeboInterface can be called in its directory which starts the interaction:

```
python gazeboInterface5_2D.py
```

(For the thesis, the 2D version is more complete than the general 3D version)

Feel free to ask me any questions at: jpoeppel@techfak.uni-bielefeld.de
