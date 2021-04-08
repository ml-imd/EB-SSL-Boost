package br.ufrn.imd.ebssb.metrics;

import weka.classifiers.evaluation.TwoClassStats;

public class ClassMetrics {

	private int classIndex;
	private String className;
	private double accuracy;
	private double recall;
	private double precision;
	private double fMeasure;
	private TwoClassStats twoClassStats;

	public ClassMetrics(int classIndex) {
		this.classIndex = classIndex;
	}

	public void getMetricsFromTwoClassStats() {
		this.accuracy = computeAccuracy();
		this.recall	= this.twoClassStats.getRecall();
		this.precision = this.twoClassStats.getPrecision();
		this.fMeasure = this.twoClassStats.getFMeasure();
	}
	
	private double computeAccuracy() {
		double acc = this.twoClassStats.getTruePositive() + this.twoClassStats.getTrueNegative();
		double acc2 = this.twoClassStats.getTruePositive() + this.twoClassStats.getFalsePositive() + this.twoClassStats.getTrueNegative() + this.twoClassStats.getFalseNegative();
		return acc / acc2;
	} 
	
	// GETTERS AND SETTERS

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
	public TwoClassStats getTwoClassStats() {
		return twoClassStats;
	}

	public void setTwoClassStats(TwoClassStats twoClassStats) {
		this.twoClassStats = twoClassStats;
		getMetricsFromTwoClassStats();
	}
	
	public String getMatrix() {
		return this.twoClassStats.getConfusionMatrix().toString(this.className);
		
	}

}
