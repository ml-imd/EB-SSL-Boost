package br.ufrn.imd.ebssb.filemanipulation;

import java.io.IOException;

public class SelfTrainingOutputWriter extends FileOutputWriter {

	private StringBuilder sb;
	
	public SelfTrainingOutputWriter(String partOfFileName) throws IOException {
		super(partOfFileName);
		sb = new StringBuilder();
	}
	
	public void logDetailsAboutStep(String datasetName, int fold) throws IOException{
		addContentline("");
		addContentline("------------------------------------------------------------------------");
		addContentline("DATASET: " + datasetName + " -> fold: " + fold);
		addContentline("------------------------------------------------------------------------");
		addContentline("");
		
		writeInFile();
	}
	
	public void printLine(String string) {
		System.out.println(string);
	}
	
	public void appendContentLine(String string) {
		sb.append(string);
		sb.append("\n");
	}
	
	public void appendContent(String string) {
		sb.append(string);
		sb.append(" ");
	}
	
	public void write() {
		System.out.println(sb.toString());
		this.sb = new StringBuilder();
	}
	
	public void outputDatasetInfo(String dataset){
		appendContentLine("");
		appendContentLine("------------------------------------------------------------------------");
		appendContentLine("Dataset: " + dataset);
		appendContentLine("------------------------------------------------------------------------");
		appendContentLine("");
		write();
	}

}
