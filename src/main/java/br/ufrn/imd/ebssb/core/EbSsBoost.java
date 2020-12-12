package br.ufrn.imd.ebssb.core;

import java.util.ArrayList;

import br.ufrn.imd.ebssb.results.FoldResult;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class EbSsBoost {

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
	
	protected ArrayList<Classifier> pool;
	protected double agreementThreshold = 75;
	
	
	
	public EbSsBoost(Dataset testSet, Dataset validationSet) {
		
		this.result = new FoldResult();
		this.history = new String();
		
		this.validationSet = new Dataset(validationSet);
		this.testSet = new Dataset(testSet);
		this.tempSet = new Dataset(testSet);
		this.tempSet.clearInstances();

		splitDatasetStratified();
		//createMainClassifier();
	}
	
	protected void splitDatasetStratified() {
		testSet.getInstances().stratify(10);
		this.labeledSet = new Dataset(testSet.getInstances().testCV(10, 0));
		this.unlabeledSet = new Dataset(testSet.getInstances().trainCV(10, 0));
	}

	protected void splitByPercentage() {
		int total = testSet.getInstances().size() * (this.labeledSetPercentual / 100);

		this.labeledSet = new Dataset(new Instances(this.testSet.getInstances(), 0, total));

		this.unlabeledSet = new Dataset(new Instances(this.testSet.getInstances(), 0, 1));
		this.unlabeledSet.clearInstances();

		for (int i = total; i < this.testSet.getInstances().size(); i++) {
			this.unlabeledSet.addInstance(this.testSet.getInstances().get(i));
		}
	}
	
	protected void trainMainCLassifierOverLabeledSet() throws Exception {
		this.mainClassifier.buildClassifier(this.labeledSet.getInstances());
	}
	
	protected void clearTempSet() {
		this.tempSet.clearInstances();
	}
	
	
}
