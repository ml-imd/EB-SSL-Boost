package br.ufrn.imd.ebssb.metrics;

public class ClassMetrics {

	private int classIndex;
	private String className;
	private double accuracy;
	private double recall;
	private double precision;
	private double fMeasure;
	
	public ClassMetrics(int classIndex) {
		this.classIndex = classIndex;
	}

	public int getClassIndex() {
		return classIndex;
	}

	public void setClassIndex(int classIndex) {
		this.classIndex = classIndex;
	}

	public String getClassName() {
		return className;
	}

	public void setClassName(String className) {
		this.className = className;
	}

	public double getAccuracy() {
		return accuracy;
	}

	public void setAccuracy(double accuracy) {
		this.accuracy = accuracy;
	}

	public double getRecall() {
		return recall;
	}

	public void setRecall(double recall) {
		this.recall = recall;
	}

	public double getPrecision() {
		return precision;
	}

	public void setPrecision(double precision) {
		this.precision = precision;
	}

	public double getfMeasure() {
		return fMeasure;
	}

	public void setfMeasure(double fMeasure) {
		this.fMeasure = fMeasure;
	}
	
}
