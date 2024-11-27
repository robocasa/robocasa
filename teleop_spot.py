from pydrake.all import ModelVisualizer, PackageMap, Simulator, StartMeshcat
from manipulation import ConfigureParser, FindResource, running_as_notebook
from manipulation.station import LoadScenario, MakeHardwareStation

# Start the visualizer.
meshcat = StartMeshcat()

# def AddSpotRemote(parser):
#     parser.package_map().AddRemote(
#         package_name="spot_description",
#         params=PackageMap.RemoteParams(
#             urls=[
#                 f"https://github.com/wrangel-bdai/spot_ros2/archive/20965ef7bba98598ee10878c7b54e6ef28a300c6.tar.gz"
#             ],
#             sha256=("20a4f12896b04cc73e186cf876bf2c7e905ee88f8add8ea51bf52dfc888674b4"),
#             strip_prefix="spot_ros2-20965ef7bba98598ee10878c7b54e6ef28a300c6/spot_description/",
#         ),
#     )

# def AddSpotRemote(parser):
#     parser.package_map().AddRemote(
#         package_name="spot_description",
#         params=PackageMap.RemoteParams(
#             urls=[
#                 f"https://github.com/wrangel-bdai/spot_ros2/archive/20965ef7bba98598ee10878c7b54e6ef28a300c6.tar.gz"
#             ],
#             sha256=("20a4f12896b04cc73e186cf876bf2c7e905ee88f8add8ea51bf52dfc888674b4"),
#             strip_prefix="spot_ros2-20965ef7bba98598ee10878c7b54e6ef28a300c6/spot_description/",
#         ),
#     )

# from most_recent_xml import most_recent_file

visualizer = ModelVisualizer(meshcat=meshcat)
ConfigureParser(visualizer.parser())
# AddSpotRemote(visualizer.parser())
visualizer.AddModels("model_2024-11-27 09:12:44.499720_no_collision.xml")
# xml_file = "../robocasa/model_2024-09-03 09:59:59.429691.xml"
# visualizer.AddModels(xml_file)
# visualizer.AddModels("mve_cabinet.xml")
# visualizer.AddModels("mve_test.xml")
# visualizer.AddModels("mmm.xml")
# visualizer.AddModels(most_recent_file) # "modified_model.xml")
# visualizer.AddModels("mve.xml")
# visualizer.AddModels("full_sample.xml")
# visualizer.AddModels("baseline_model.xml")
# visualizer.AddModels(
#     url="package://manipulation/spot/spot_with_arm_and_floating_base_actuators.urdf"
# )
# visualizer.AddModels(
#     url="package://manipulation/spot/spot_with_arm.urdf"
# )
visualizer.Run(loop_once=0)
meshcat.DeleteAddedControls()

input("Press enter to exit. ")
