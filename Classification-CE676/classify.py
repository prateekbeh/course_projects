"""
Steps to Run Code:-

1.) Set the path vaiable to the folder which contains all the .csv files of Ground, House ,,Trees.
2.) The files should be CSV files and not space seperated.
3.) Remove all the headers and no. of row count from top of the file.
4.) The names of file should be starting with House*.csv, Ground*.csv , Tree*.csv (Please take care of upper case , lower case letters , otherwise the file will not be read)

"""



import os
import pandas as pd

path = "/Users/behprat/Downloads/CE676/Assignment-1/segmented_data/"

fout = open(path + "../classified.csv","w");


def classify(fh,fout,flag):
	data = fh.readline();
	while(data):
		if not data:
			break;
		coord = data.split(",");
		if len(coord) != 9:
			continue;
		classified_list = coord[0] + "," + coord[1] + "," + coord[2] + "," + flag;
		fout.write(classified_list + '\n');
		data = fh.readline();

def remove_duplicates(in_path):
	data = pd.read_csv(in_path)
	data_unique = data.drop_duplicates(subset=['X','Y','Z'], keep='first')
	return data_unique

def mark_unclassified(marked_df , orig_data_path):
	orig_data = pd.read_csv(orig_data_path, usecols=['X','Y','Z'])
	tmp_df = marked_df[['X','Y','Z']];
	unclassified_data = orig_data[~orig_data.isin(tmp_df).all(1)]

if os.path.exists(path):
	fout.write("X,Y,Z,Class" + "\n");
	for files in os.listdir(path):
		if files.endswith(".csv"):
			print(path + files)
			fh = open(path + files,'r');
			if files.startswith("Trees"):
				classify(fh,fout,"2")
			if files.startswith("House"):
				classify(fh,fout,"3")
			if files.startswith("Ground"):
				classify(fh,fout,"1")
			fh.close();
	
	#df_unique = remove_duplicates(path + "classified.csv")
	
	#total_classfied_data = mark_unclassified(df_unique,path + "total.csv")

fout.close();
