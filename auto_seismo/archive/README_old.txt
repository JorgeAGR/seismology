* This set of python scripts will allow to:
a) Arrange a set of seismograms into a set of arrays sliced at a 40 second interval 
   containing the theoretical arrival of a chosen wave. These are then converted into
   binary objects for ease of use and light size.
b) Using the above binary arrays, they are loaded onto a new script that will initialize
   and train a random number of CNN models (set to 10 by default) in order to find the
   best converging system to predict the arrival for desired seismograms.

Instructions:

 - Open the 'cnn_config.txt' file and input the desired directories in the appropriate variable.
   Seperate each directory with '||', eg: ../sample/dir0/||../sample/dir1/.
   Input then the desired wave to predict by inputing it's appropriate SAC variable that holds the
   theoretically calculated value.
   
   *!* Always end directories with a forwardslash '/'
   *!* The scripts are ideally executed wtih the '*_exe' bash files in the main folder (where this
       README) is, so that will be the working directory for all other directory references.

 # The steps taken to make it work are kind of rudimentary for now until full implementation of libraries
   in the Styx cluster can be achieved.

 - Have a computer with SAC to run the 'arrange_exe' script. This will format all the files in the directories
   entered in the config file to the appropriate binary array.

 - Compress the entire auto_seismo folder and send it to a computer (preferrably a cluster with GPU).
   Unpack the folder again, and this time run the 'pred_exe' file.

   *!* Two files are included in the 'job_submit/' folder that allow running the above script as a job in the
   NMSU cluster (Discovery aka Joker). A GPU and CPU version are provided.

 - The name of the file and its corresponding prediction will then be saved in a CSV file in the 'results/' folder.