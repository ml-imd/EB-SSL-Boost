package br.ufrn.imd.ebssb.core;

import weka.classifiers.Classifier;

public class ClassifierWithInfo {

	protected Classifier classifier;
	protected double weight;
	protected int hits;
	
	public ClassifierWithInfo(Classifier classifier) {
		this.classifier = classifier;
		this.weight = 1.0;
	}
	
	public Classifier getClassifier() {
		return classifier;
	}
	
	public double getWeight() {
		return weight;
	}
	
	public void setWeight(double weight) {
		this.weight = weight;
	}
	
	public double getHits() {
		return hits;
	}
	
	public void resetHits() {
		hits = 0;
	}
	
	public void countAccuracy(double output, double real) {
		if(output == real) {
			hits += 1;
		}
	}
}
