package br.ufrn.imd.ebssb.core;

import java.io.Serializable;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instances;

public class Measures implements Serializable {

	public void sum(Measures b) {
		for (int i = 0; i < precision.length; i++) {
			precision[i] += b.precision[i];
			recall[i] += b.recall[i];
			fmeasure[i] += b.fmeasure[i];
		}
	}

	private static final long serialVersionUID = 1L;

	private double[] precision;
	private double[] recall;
	private double[] fmeasure;
	private double[] classesDistribution;
	private String[] labels;
	private double accuracy;
	private double error;
	
	public Measures(Classifier classifier, Instances train, Instances test) throws Exception {
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(classifier, test);
		init(eval, test);
	}

	public Measures(Classifier classifier, Instances instances, int numFolds, Random rand) throws Exception {
		Evaluation eval = new Evaluation(instances);
		eval.crossValidateModel(classifier, instances, numFolds, rand);
		init(eval, instances);
	}

	private void init(Evaluation eval, Instances instances) {
		int numClasses = instances.numClasses();
		double numInstances = instances.numInstances();
		precision = new double[numClasses];
		recall = new double[numClasses];
		fmeasure = new double[numClasses];
		accuracy = eval.pctCorrect();
		error = eval.pctIncorrect();
		
		classesDistribution = eval.getClassPriors();
		for (int i = 0; i < numClasses; i++) {
			precision[i] = correctValue(eval.precision(i));
			recall[i] = correctValue (eval.recall(i));
			fmeasure[i] = correctValue (eval.fMeasure(i));
			classesDistribution[i] /= numInstances;
		}

		Attribute att = instances.classAttribute();
		labels = new String[numClasses];
		for (int i = 0; i < numClasses; i++) {
			labels[i] = att.value(i);
		}
	}
	
	private double correctValue (double value) {
		return Double.isNaN(value) ? 0 : value;
	}

	public double[] precision() {
		return precision;
	}

	public double[] recall() {
		return recall;
	}

	public double[] fMeasure() {
		return fmeasure;
	}

	public double getPrecisionMean() {
		return average(precision);
	}

	public double getRecallMean() {
		return average(recall);
	}

	public double getFmeasureMean() {
		double pavg = getPrecisionMean();
		double ravg = getRecallMean();
		return fMeasure(pavg, ravg);
	}

	public double precisionWeightedMean() {
		return averageByDistribution(precision);
	}

	public double recallWeightedMean() {
		return averageByDistribution(recall);
	}

	public double fMeasureWeightedMean() {
		double pwavg = precisionWeightedMean();
		double rwavg = recallWeightedMean();
		return fMeasure(pwavg, rwavg);
	}

	public String toSummary() {
		int maxLength = 15;
		for (String str : labels) {
			if (str.length() > maxLength) {
				maxLength = str.length() + 2;
			}
		}

		double[][] values = { precision(), recall(), fMeasure() };

		String mask = "  %" + maxLength + "s";
		String maskValue = "  %" + maxLength + ".4f";

		StringBuilder str = new StringBuilder(2000);
		String bigmask = mask + maskValue + maskValue + maskValue + "\n";
		str.append(String.format(mask + mask + mask + mask + "\n", "Class", "Precision", "Recall", "F-Measure"));
		for (int i = 0; i < labels.length; i++) {
			str.append(String.format(bigmask, labels[i], values[0][i], values[1][i], values[2][i]));
		}
		str.append("\n");
		str.append(String.format(bigmask, "Simple AVG", getPrecisionMean(), getRecallMean(), getFmeasureMean()));
		str.append(String.format(bigmask, "Weighted AVG", precisionWeightedMean(), recallWeightedMean(),
				fMeasureWeightedMean()));

		return str.toString();
	}

	private double fMeasure(double precision, double recall) {
		return (2.0 * (precision * recall)) / (precision + recall);
	}

	private double average(double[] values) {
		double avg = 0;
		for (double d : values) {
			avg += d;
		}
		return avg / (double) values.length;
	}

	private double averageByDistribution(double[] values) {
		double avg = 0;
		for (int i = 0; i < values.length; i++) {
			avg += values[i] * classesDistribution[i];
		}
		return avg;
	}

	public double getAccuracy() {
		return accuracy;
	}

	public void setAccuracy(double accuracy) {
		this.accuracy = accuracy;
	}

	public double getError() {
		return error;
	}

	public void setError(double error) {
		this.error = error;
	}
	
	
}
