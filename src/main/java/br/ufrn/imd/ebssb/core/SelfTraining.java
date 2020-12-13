package br.ufrn.imd.ebssb.core;

import br.ufrn.imd.ebssb.results.FoldResult;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

public class SelfTraining {

	protected Dataset validationSet;
	protected Dataset testSet;

	protected Dataset labeledSet;
	protected Dataset unlabeledSet;
	protected int labeledSetPercentual = 10;

	protected int unlabeledSetJoinRate = 10;
	protected int amountToJoin = 0;
	
	protected Dataset tempSet;

	protected Classifier mainClassifier;

	protected int goodClassifiedInstances = 0;
	protected int missClassifiedInstances = 0;
	
	protected FoldResult result;
	protected String history;
	protected String iterationInfo;
	
	public SelfTraining() {
		
	}
	
	public SelfTraining(Dataset testSet, Dataset validationSet) {
		
		this.result = new FoldResult();
		this.history = new String();
		
		this.validationSet = new Dataset(validationSet);
		this.testSet = new Dataset(testSet);
		this.tempSet = new Dataset(testSet);
		this.tempSet.getInstances().clear();

		splitDatasetStratified();
		createMainClassifier();
	}
	
	protected void splitDatasetStratified() {
		testSet.getInstances().stratify(10);
		this.labeledSet = new Dataset(testSet.getInstances().testCV(10, 0));
		this.unlabeledSet = new Dataset(testSet.getInstances().trainCV(10, 0));
	}

	
	
	protected void trainMainCLassifierOverLabeledSet() throws Exception {
		this.mainClassifier.buildClassifier(this.labeledSet.getInstances());
	}
	
	protected void clearTempSet() {
		this.tempSet.getInstances().clear();
	}
	
	protected void createMainClassifier() {
		// weka.classifiers.trees.J48 -C 0.05 -M 2 (74.4792)
		/*
		 * J48 j48 = new J48(); try {
		 * j48.setOptions(weka.core.Utils.splitOptions("-C 0.05 -M 2")); } catch
		 * (Exception e) { e.printStackTrace(); } this.mainClassifier = (J48) j48;
		 */
		
		//new for test - 26/01/2020
		//weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a
		
		MultilayerPerceptron mlp = new MultilayerPerceptron();
		try {
			mlp.setOptions(weka.core.Utils.splitOptions("-L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a"));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		this.mainClassifier = mlp;
	}
	
	protected void joinClassifiedWithLabeledSet() {
		labeledSet.getInstances().addAll(this.tempSet.getInstances());
	}
	
	protected void mainClassifierJob() throws Exception{
		Measures measures = new Measures(this.mainClassifier, this.labeledSet.getInstances(), this.validationSet.getInstances());
		
		result.setAccuracy(measures.getAccuracy());
		result.setError(measures.getError());
		result.setfMeasure(measures.getFmeasureMean());
		result.setPrecision(measures.getPrecisionMean()); 
		result.setRecall(measures.getRecallMean()); 	
	}
	
	protected void printIterationInfo() {
		System.out.println(this.iterationInfo);
	}
	
	protected void addIterationInfoToHistory() {
		addToHistory(this.iterationInfo);
	}
	
	protected void addToHistory(String string) {
		StringBuilder sb = new StringBuilder();
		sb.append(this.history);
		sb.append(string);
		
		this.history = new String(sb.toString());
	}
	
	protected void generateIterationInfo(int iteration) {
		
		StringBuilder sb = new StringBuilder();
		
		sb.append("--------------------------------------------------\n");
		sb.append("@ SELFTRAINING ITERATION: " + iteration + "\n");
		sb.append("--------------------------------------------------\n");
		sb.append("\n");
		sb.append("@ LABELED: " + this.labeledSet.getInstances().size() + "\n");
		sb.append("@ UNLABELED: " + this.unlabeledSet.getInstances().size() + "\n");
		if(iteration > 1) {
			sb.append("@ From unlabeled to labeled (last iteration): " + goodClassifiedInstances + "\n");
		}
		sb.append("--------------------------------------------------\n");
		this.iterationInfo = new String(sb.toString());
	}
	
	//GETTERS AND SETTERS 
	public Dataset getValidationSet() {
		return validationSet;
	}

	public void setValidationSet(Dataset validationSet) {
		this.validationSet = validationSet;
	}

	public Dataset getTestSet() {
		return testSet;
	}

	public void setTestSet(Dataset testSet) {
		this.testSet = testSet;
	}

	public Dataset getLabeledSet() {
		return labeledSet;
	}

	public void setLabeledSet(Dataset labeledSet) {
		this.labeledSet = labeledSet;
	}

	public Dataset getUnlabeledSet() {
		return unlabeledSet;
	}

	public void setUnlabeledSet(Dataset unlabeledSet) {
		this.unlabeledSet = unlabeledSet;
	}

	public Dataset getTempSet() {
		return tempSet;
	}

	public void setTempSet(Dataset tempSet) {
		this.tempSet = tempSet;
	}

	public Classifier getMainClassifier() {
		return mainClassifier;
	}

	public void setMainClassifier(Classifier mainClassifier) {
		this.mainClassifier = mainClassifier;
	}

	public int getGoodClassifiedInstances() {
		return goodClassifiedInstances;
	}

	public void setGoodClassifiedInstances(int goodClassifiedInstances) {
		this.goodClassifiedInstances = goodClassifiedInstances;
	}

	public FoldResult getResult() {
		return result;
	}

	public void setResult(FoldResult result) {
		this.result = result;
	}

	public String getHistory() {
		return history;
	}

	public void setHistory(String history) {
		this.history = history;
	}

	public String getIterationInfo() {
		return iterationInfo;
	}

	public void setIterationInfo(String iterationInfo) {
		this.iterationInfo = iterationInfo;
	}

	public int getLabeledSetPercentual() {
		return labeledSetPercentual;
	}

	public void setLabeledSetPercentual(int labeledSetPercentual) {
		this.labeledSetPercentual = labeledSetPercentual;
	}

	public int getUnlabeledSetJoinRate() {
		return unlabeledSetJoinRate;
	}

	public void setUnlabeledSetJoinRate(int unlabeledSetJoinRate) {
		this.unlabeledSetJoinRate = unlabeledSetJoinRate;
	}

	public int getAmountToJoin() {
		return amountToJoin;
	}

	public void setAmountToJoin(int amountToJoin) {
		this.amountToJoin = amountToJoin;
	}

	public int getMissClassifiedInstances() {
		return missClassifiedInstances;
	}

	public void setMissClassifiedInstances(int missClassifiedInstances) {
		this.missClassifiedInstances = missClassifiedInstances;
	}
}
