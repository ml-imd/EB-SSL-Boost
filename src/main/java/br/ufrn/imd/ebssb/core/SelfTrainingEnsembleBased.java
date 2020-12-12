package br.ufrn.imd.ebssb.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import br.ufrn.imd.ebssb.results.InstanceResult;
import br.ufrn.imd.ebssb.utils.Mathematics;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;

public class SelfTrainingEnsembleBased extends SelfTraining{

	protected ArrayList<Classifier> pool;
	protected double agreementThreshold = 75; //percentual
	
	public SelfTrainingEnsembleBased(Dataset testSet, Dataset validationSet) {
		super(testSet, validationSet);
		this.pool = new ArrayList<Classifier>();
		populatePool();
	}

	/**
	 * make pseudo-label with pool and after check the agreement, get the class for the
	 * best instances (above agreementThreshold) with the main classifier) 
	 * 
	 * @throws Exception
	 */
	public void runVersionOne() throws Exception {
		
		int i = 1;
		while (true) {
			generateIterationInfo(i);
			addIterationInfoToHistory();
			
			trainMainCLassifierOverLabeledSet();
			trainClassifiersPool();
			classifyInstancesAndCheckAgreement(this.unlabeledSet);

			if (tempSet.getInstances().size() == 0) {
				break;
			}
			
			classifyBestWithMainClassifier();
			joinClassifiedWithLabeledSet();
			result.addIterationInfo(this.goodClassifiedInstances, this.missClassifiedInstances);
			clearTempSet();
			i++;
			printIterationInfo();
		}
		mainClassifierJob();
	}

	/**
	 * make pseudo-label with pool and after check the agreement, get the class
	 * for the best instances (above agreementThreshold) using the label voted by pool
	 * 
	 * @throws Exception
	 */
	public void runVersionTwo() throws Exception {
		int i = 1;
		while (true) {
			generateIterationInfo(i);
			trainMainCLassifierOverLabeledSet();
			trainClassifiersPool();
			classifyInstancesCheckAgreementPool(this.unlabeledSet);

			if (tempSet.getInstances().size() == 0) {
				break;
			}
			joinClassifiedWithLabeledSet();
			result.addIterationInfo(this.goodClassifiedInstances, this.missClassifiedInstances);
			clearTempSet();
			i++;
			printIterationInfo();
			addIterationInfoToHistory();
		}
		mainClassifierJob();
	}
	
	/**
	 * TO DO
	 * 
	 * @throws Exception
	 */
	public void runVersionOneDistanceFactor() throws Exception {
		
		this.amountToJoin = this.unlabeledSet.getInstances().size() / this.unlabeledSetJoinRate;
		
		int i = 1;
		while (true) {
			generateIterationInfo(i);
			addIterationInfoToHistory();
			
			trainMainCLassifierOverLabeledSet();
			trainClassifiersPool();
			classifyInstancesAndCheckAgreementDistanceFactor(this.unlabeledSet);

			if (tempSet.getInstances().size() == 0) {
				break;
			}
			
			classifyBestWithMainClassifier();
			joinClassifiedWithLabeledSet();
			result.addIterationInfo(this.goodClassifiedInstances, this.missClassifiedInstances);
			clearTempSet();
			i++;
			printIterationInfo();
		}
		mainClassifierJob();
	}
	
	/**
	 * TO DO
	 * 
	 * @throws Exception
	 */
	public void runVersionTwoDistanceFactor() throws Exception {
		
		this.amountToJoin = this.unlabeledSet.getInstances().size() / this.unlabeledSetJoinRate;
		
		int i = 1;
		while (true) {
			generateIterationInfo(i);
			addIterationInfoToHistory();
			trainMainCLassifierOverLabeledSet();
			trainClassifiersPool();
			classifyInstancesCheckAgreementPoolDistanceFactor(this.unlabeledSet);

			if (tempSet.getInstances().size() == 0) {
				break;
			}
			joinClassifiedWithLabeledSet();
			result.addIterationInfo(this.goodClassifiedInstances, this.missClassifiedInstances);
			clearTempSet();
			i++;
			printIterationInfo();
			
		}
		mainClassifierJob();
	}
	
	
	public void trainClassifiersPool() throws Exception {

		for (Classifier c: pool) {
			c.buildClassifier(this.labeledSet.getInstances());
		}

	}

	private void classifyInstancesAndCheckAgreement(Dataset dataset) throws Exception {
		
		StringBuilder sb = new StringBuilder();
		sb.append("UNLABELED SET ITERATION RESULT: \n\n");
		
		double agreementValue = pool.size() * agreementThreshold / 100;
		int value = (int) agreementValue;
		
		InstanceResult instanceResult;
		
		Iterator<Instance> iterator = this.unlabeledSet.getInstances().iterator();
		while(iterator.hasNext()) {
			Instance instance = iterator.next();
			instanceResult = new InstanceResult(instance);
			for(Classifier c: this.pool) {
				instanceResult.addPrediction(c.classifyInstance(instance));
			}
			
			if(instanceResult.getBestAgreement() >= value) {
				tempSet.addInstance((Instance)instance.copy()); //CAUTION
				iterator.remove();
			}
			sb.append(instanceResult.outputDataToCsv() + "\n");
		}
		this.goodClassifiedInstances = tempSet.getInstances().size();
		sb.append("\n");
		addToHistory(sb.toString());
	}
	
	private void classifyInstancesCheckAgreementPool(Dataset dataset) throws Exception {
		
		this.missClassifiedInstances = 0;
		
		StringBuilder sb = new StringBuilder();
		sb.append("UNLABELED SET ITERATION RESULT: \n\n");
		
		double agreementValue = pool.size() * agreementThreshold / 100;
		int value = (int) agreementValue;
		
		InstanceResult instanceResult;
		
		Iterator<Instance> iterator = this.unlabeledSet.getInstances().iterator();
		while(iterator.hasNext()) {
			Instance instance = iterator.next();
			instanceResult = new InstanceResult(instance);
			for(Classifier c: this.pool) {
				instanceResult.addPrediction(c.classifyInstance(instance));
			}
			
			if(instanceResult.getBestAgreement() >= value) {
				DenseInstance d = (DenseInstance) instance.copy();
				d.setClassValue(instanceResult.getBestClass());
				
				if(d.classValue() != instance.classValue()) {
					this.missClassifiedInstances += 1;
				}
				
				tempSet.addInstance(d); //CAUTION
				iterator.remove();
			}
			sb.append(instanceResult.outputDataToCsv() + "\n");
		}
		this.goodClassifiedInstances = tempSet.getInstances().size();
		sb.append("\n");
		addToHistory(sb.toString());
	}
	
	private void classifyInstancesAndCheckAgreementDistanceFactor(Dataset dataset) throws Exception {
		
		StringBuilder sb = new StringBuilder();
		sb.append("UNLABELED SET ITERATION RESULT: \n\n");
		
		ArrayList<InstanceResult> standardResults = new ArrayList<InstanceResult>();
		Instance[] centroids =  Mathematics.centroidsOf(this.labeledSet.getInstances());
		int amount = this.amountToJoin;

		InstanceResult instanceResult;
		
		Iterator<Instance> iterator = this.unlabeledSet.getInstances().iterator();
		while(iterator.hasNext()) {
			Instance instance = iterator.next();
			instanceResult = new InstanceResult(instance);
			for(Classifier c: this.pool) {
				instanceResult.addPrediction(c.classifyInstance(instance));
			}
			
			Double distance = Mathematics.euclidianDistance(instance, centroids[instanceResult.getBestClassIndex()]);
			instanceResult.setFactor(instanceResult.getBestAgreement() * (1 / distance));
			standardResults.add(instanceResult);
		}
			
		Collections.sort(standardResults, InstanceResult.factorComparatorDesc);
		
		if(this.unlabeledSet.getInstances().size() < amount*2) {
			amount = this.unlabeledSet.getInstances().size();
		}
		
		for(InstanceResult ir: standardResults) {
			sb.append(ir.outputDataToCsvWithDistanceFactor() + "\n");
		}
		
		for(int i = 0; i < amount; i++) {
			DenseInstance d = (DenseInstance) standardResults.get(i).getInstance().copy();
			//class value come from main classifier in next step (there was here and line to get best class and put inside class of "d")
			tempSet.addInstance(d); //CAUTION
			unlabeledSet.getInstances().remove(standardResults.get(i).getInstance());
		}
		
		this.goodClassifiedInstances = tempSet.getInstances().size();
		sb.append("\n");
		addToHistory(sb.toString());
	}
	
	private void classifyInstancesCheckAgreementPoolDistanceFactor(Dataset dataset) throws Exception {
		
		StringBuilder sb = new StringBuilder();
		sb.append("UNLABELED SET ITERATION RESULT: \n\n");
		
		ArrayList<InstanceResult> standardResults = new ArrayList<InstanceResult>();
		Instance[] centroids =  Mathematics.centroidsOf(this.labeledSet.getInstances());
		int amount = this.amountToJoin;
		
		InstanceResult instanceResult;
		
		Iterator<Instance> iterator = this.unlabeledSet.getInstances().iterator();
		while(iterator.hasNext()) {
			Instance instance = iterator.next();
			instanceResult = new InstanceResult(instance);
			for(Classifier c: this.pool) {
				instanceResult.addPrediction(c.classifyInstance(instance));
			}
			
			Double distance = Mathematics.euclidianDistance(instance, centroids[instanceResult.getBestClassIndex()]);
			instanceResult.setFactor(instanceResult.getBestAgreement() * (1 / distance));
			standardResults.add(instanceResult);
		}
		
		Collections.sort(standardResults, InstanceResult.factorComparatorDesc);
		
		if(this.unlabeledSet.getInstances().size() < amount*2) {
			amount = this.unlabeledSet.getInstances().size();
		}
		
		for(InstanceResult ir: standardResults) {
			sb.append(ir.outputDataToCsvWithDistanceFactor() + "\n");
		}
		
		for(int i = 0; i < amount; i++) {
			DenseInstance d = (DenseInstance) standardResults.get(i).getInstance().copy();
			d.setClassValue(standardResults.get(i).getBestClass());
			tempSet.addInstance(d); //CAUTION
			unlabeledSet.getInstances().remove(standardResults.get(i).getInstance());
		}
		
		this.goodClassifiedInstances = tempSet.getInstances().size();
		sb.append("\n");
		addToHistory(sb.toString());
	}
	
	private void classifyBestWithMainClassifier() throws Exception {
		this.missClassifiedInstances = 0;
		for (Instance i : this.tempSet.getInstances()) {
			double x = this.mainClassifier.classifyInstance(i);
			
			if(x != i.classValue()) {
				this.missClassifiedInstances += 1;
			}
			
			i.setClassValue(x);
		}
	}

	public void populatePool() {
		J48 j48a = new J48();
		J48 j48b = new J48();
		J48 j48c = new J48();
		J48 j48d = new J48();

		NaiveBayes nb1 = new NaiveBayes();
		NaiveBayes nb2 = new NaiveBayes();
		NaiveBayes nb3 = new NaiveBayes();

		IBk ibk1 = new IBk();
		IBk ibk2 = new IBk();
		IBk ibk3 = new IBk();
		IBk ibk4 = new IBk();
		IBk ibk5 = new IBk();

		SMO smo1 = new SMO();
		SMO smo2 = new SMO();
		SMO smo3 = new SMO();
		SMO smo4 = new SMO();
		SMO smo5 = new SMO();

		DecisionTable dt1 = new DecisionTable();
		DecisionTable dt2 = new DecisionTable();
		DecisionTable dt3 = new DecisionTable();

		try {

			j48a.setOptions(weka.core.Utils.splitOptions("-C 0.05 -M 2"));
			j48b.setOptions(weka.core.Utils.splitOptions("-C 0.10 -M 2"));
			j48c.setOptions(weka.core.Utils.splitOptions("-C 0.20 -M 2"));
			j48d.setOptions(weka.core.Utils.splitOptions("-C 0.25 -M 2"));

			nb1.setOptions(weka.core.Utils.splitOptions(""));
			nb2.setOptions(weka.core.Utils.splitOptions("-K"));
			nb3.setOptions(weka.core.Utils.splitOptions("-D"));

			ibk1.setOptions(weka.core.Utils.splitOptions(
					"-K 1 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -R first-last\\\"\""));
			ibk2.setOptions(weka.core.Utils.splitOptions(
					"-K 3 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -R first-last\\\"\""));
			ibk3.setOptions(weka.core.Utils.splitOptions(
					"-K 3 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.ManhattanDistance -R first-last\\\"\""));
			ibk4.setOptions(weka.core.Utils.splitOptions(
					"-K 5 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -R first-last\\\"\""));
			ibk5.setOptions(weka.core.Utils.splitOptions(
					"-K 5 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.ManhattanDistance -R first-last\\\"\""));

			smo1.setOptions(weka.core.Utils.splitOptions(
					"-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""));
			smo2.setOptions(weka.core.Utils.splitOptions(
					"-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.NormalizedPolyKernel -E 2.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""));
			smo3.setOptions(weka.core.Utils.splitOptions(
					"-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""));
			smo4.setOptions(weka.core.Utils.splitOptions(
					"-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.Puk -O 1.0 -S 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""));
			smo5.setOptions(weka.core.Utils.splitOptions(
					"-C 0.8 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""));

			dt1.setOptions(weka.core.Utils.splitOptions("-X 1 -S \"weka.attributeSelection.BestFirst -D 1 -N 5\""));
			dt2.setOptions(weka.core.Utils.splitOptions("-X 1 -S \"weka.attributeSelection.BestFirst -D 1 -N 3\""));
			dt3.setOptions(weka.core.Utils.splitOptions("-X 1 -S \"weka.attributeSelection.BestFirst -D 1 -N 7\""));

		} catch (Exception e) {
			e.printStackTrace();
		}

		this.pool.add(j48a);
		this.pool.add(j48b);
		this.pool.add(j48c);
		this.pool.add(j48d);

		this.pool.add(nb1);
		this.pool.add(nb2);
		this.pool.add(nb3);

		this.pool.add(ibk1);
		this.pool.add(ibk2);
		this.pool.add(ibk3);
		this.pool.add(ibk4);
		this.pool.add(ibk5);

		this.pool.add(smo1);
		this.pool.add(smo2);
		this.pool.add(smo3);
		this.pool.add(smo4);
		this.pool.add(smo5);

		this.pool.add(dt1);
		this.pool.add(dt2);
		this.pool.add(dt3);
	}

	//GETTERS AND SETTERS
 	public ArrayList<Classifier> getPool() {
		return pool;
	}

	public void setPool(ArrayList<Classifier> pool) {
		this.pool = pool;
	}

	public double getAgreementThreshold() {
		return agreementThreshold;
	}

	public void setAgreementThreshold(double agreementThreshold) {
		this.agreementThreshold = agreementThreshold;
	}
	
}
