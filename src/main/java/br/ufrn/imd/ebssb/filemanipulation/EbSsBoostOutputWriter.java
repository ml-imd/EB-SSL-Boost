package br.ufrn.imd.ebssb.filemanipulation;

import java.io.IOException;

public class EbSsBoostOutputWriter extends FileOutputWriter {
	
	public EbSsBoostOutputWriter(String partOfFileName) throws IOException {
		super(partOfFileName);
	}
	
	public void logGeneralDetails(String datasetName) throws IOException{
		addContentLine("");
		addContentLine("================================================================================================");        
		addContentLine("DATASET: " + datasetName);
		addContentLine("================================================================================================");
		
		writeInFile();
	}
	
	
	public void logDetailsAboutStep(String datasetName, int fold) throws IOException{
		addContentLine("");
		addContentLine("================================================================================================");
		addContentLine("DATASET: " + datasetName + " -> fold: " + fold);
		addContentLine("================================================================================================");
		
		writeInFile();
	}
	
	public void printLine(String string) {
		System.out.println(string);
	}
		
	public void printContent() {
		System.out.println(toText.toString());
	}
	
	public void outputDatasetInfo(String dataset){
		addContentLine("");
		addContentLine("================================================================================================");
		addContentLine("Dataset: " + dataset);
		addContentLine("================================================================================================");
		printContent();
	}
	
	

}
