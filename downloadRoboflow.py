import roboflow

rf = roboflow.Roboflow(api_key="3piPof3nd2m82YCCLDD7")
project = rf.workspace().project("netrasahaya-8yhey")
model = project.version(1).download("tensorflow")
