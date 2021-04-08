package br.ufrn.imd.ebssb.metrics;

import java.util.ArrayList;

import br.ufrn.imd.ebssb.core.Dataset;
import weka.classifiers.evaluation.ConfusionMatrix;
import weka.classifiers.evaluation.NominalPrediction;

public class Prediction {

	private ConfusionMatrix matrix;
	private Dataset dataset;
	private ArrayList<ClassMetrics> classesMetrics;

	private int totalIntances = 0;

	public Prediction(Dataset d) {
		this.classesMetrics = new ArrayList<ClassMetrics>();
		this.dataset = new Dataset(d);
		buildClassNames();
		this.matrix = new ConfusionMatrix(getClassNamesAsArray());
	}

	public void addPrediction(double actual, double predicted) {
		NominalPrediction np = new NominalPrediction(actual, NominalPrediction.makeDistribution(predicted, 3));
		try {
			this.matrix.addPrediction(np);
			this.totalIntances++;
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void buildClassNames() {
		int numClasses = this.dataset.getInstances().numClasses();

		ClassMetrics metrics;
		for (int i = 0; i < numClasses; i++) {
			metrics = new ClassMetrics(i);
			metrics.setClassName(this.dataset.getInstances().classAttribute().value(i));
			classesMetrics.add(metrics);
		}
	}

	private String[] getClassNamesAsArray() {
		String[] ss = new String[classesMetrics.size()];
		for (int i = 0; i < classesMetrics.size(); i++) {
			ss[i] = classesMetrics.get(i).getClassName();
		}
		return ss;
	}

	public void buildMetrics() {
		for(int i = 0; i < classesMetrics.size(); i++) {
			classesMetrics.get(i).setTwoClassStats(matrix.getTwoClassStats(i));
		}
	}
	
	//GETTERS AND SETTERS
	
	public ConfusionMatrix getMatrix() {
		return matrix;
	}

	public void setMatrix(ConfusionMatrix matrix) {
		this.matrix = matrix;
	}

	public Dataset getDataset() {
		return dataset;
	}

	public void setDataset(Dataset dataset) {
		this.dataset = dataset;
	}

	public ArrayList<ClassMetrics> getClassesMetrics() {
		return classesMetrics;
	}

	public void setClassesMetrics(ArrayList<ClassMetrics> classesMetrics) {
		this.classesMetrics = classesMetrics;
	}

	public int getTotalIntances() {
		return totalIntances;
	}

	public void setTotalIntances(int totalIntances) {
		this.totalIntances = totalIntances;
	}


}
