class Params():
	def __init__(self, filename):
		ifile = open(filename, 'r')

		line = ifile.readline()

		self.save_dir = ""
		self.model_name = ""
		self.restore_file = ""

		while(len(line) != 0):
			if(line[0] != '#' and line[0]!='\n'):
				split = line.split()
				if(len(split) == 3):
					if(split[0] == "ROS_PKG_NAME"):
						self.ros_pkg_name = split[2]

					if(split[0] == "CHECKPOINT_DIRECTORY"):
						self.save_dir = split[2]
					if(split[0] == "CHECKPOINT_NAME"):
						self.model_name = split[2]
					if(split[0] == "IRNV2_CHECKPOINT_DIR"):
						self.irnv2_checkpoint_dir = split[2]
					if(split[0] == "IRNV2_CHECKPOINT"):
						self.irnv2_checkpoint = split[2]
						
					if(split[0] == "RESTORE_CHECKPOINT"):
						self.restore_file = split[2]
					if(split[0] == "LOG_DIR"):
						self.log_dir = split[2]
						
					if(split[0] == "TRAIN_DIR"):
						self.train_dir = split[2]
					if(split[0] == "TEST_DIR"):
						self.test_dir = split[2]

			line = ifile.readline()
		
if __name__ == '__main__':
	p = Params("params_file.txt")