package br.ufrn.imd.ebssb.metrics;

import java.io.Serializable;

import weka.classifiers.Evaluation;
import weka.core.Attribute;

public class Measures implements Serializable {

	private static final long serialVersionUID = 1L;

	private double[] precision;
	private double[] recall;
	private double[] fmeasure;
	private double[] classesDistribution;
	private String[] labels;
	private double accuracy;
	private double error;

	public Measures(Prediction prediction) throws Exception {
		Evaluation eval = new Evaluation(prediction.getDataset().getInstances());
		init(eval, prediction);
	}

	private void init(Evaluation eval, Prediction prediction) {
		int numClasses = prediction.getDataset().getInstances().numClasses();
		double numInstances = prediction.getDataset().getInstances().numInstances();

		precision = new double[numClasses];
		recall = new double[numClasses];
		fmeasure = new double[numClasses];

		accuracy = calcAccuracy(prediction.getMatrix().correct(), prediction.getDataset().getInstances().size());
		error = calcError(prediction.getMatrix().incorrect(), prediction.getDataset().getInstances().size());

		classesDistribution = eval.getClassPriors();
		for (int i = 0; i < numClasses; i++) {
			precision[i] = correctValue(prediction.getClassesMetrics().get(i).getPrecision());
			recall[i] = correctValue(prediction.getClassesMetrics().get(i).getRecall());
			fmeasure[i] = correctValue(prediction.getClassesMetrics().get(i).getRecall());
			classesDistribution[i] /= (numInstances + numClasses);
		}

		Attribute att = prediction.getDataset().getInstances().classAttribute();
		labels = new String[numClasses];
		for (int i = 0; i < numClasses; i++) {
			labels[i] = att.value(i);
		}
	}

	private double calcAccuracy(double totalCorrect, int totalInstances) {
		double correct = totalCorrect;
		double total = totalInstances;
		return (correct / total) * 100;
	}

	private double calcError(double totalncorrect, int totalInstances) {
		double incorrect = totalncorrect;
		double total = totalInstances;
		return (incorrect / total) * 100;
	}

	private double correctValue(double value) {
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

	public void sum(Measures b) {
		for (int i = 0; i < precision.length; i++) {
			precision[i] += b.precision[i];
			recall[i] += b.recall[i];
			fmeasure[i] += b.fmeasure[i];
		}
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

	// GETTER E SETTERS

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

	/*
	 * Evaluation eval = new Evaluation(validation.getInstances());
	 * System.out.println("----------------------"); for(Instance inst:
	 * validation.getInstances()) { System.out.println(inst.weight() + " -- " +
	 * inst.classValue()); } System.out.println("----------------------");
	 * System.out.println(validation.getMyInstances().size());
	 * System.out.println(validation.getInstances().size()); double[] d =
	 * eval.getClassPriors(); for(int z = 0; z < d.length; z++) {
	 * System.out.println( d[z] ); } break;
	 */

}
