package br.ufrn.imd.ebssb.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import br.ufrn.imd.ebssb.results.InstanceResultStandard;
import br.ufrn.imd.ebssb.utils.Mathematics;
import weka.core.DenseInstance;
import weka.core.Instance;

public class SelfTrainingStandard extends SelfTraining{
	
	public SelfTrainingStandard(Dataset testSet, Dataset validationSet) {
		super(testSet, validationSet);
	}

	/**
	 * Join the p percent best instances (unlabeledSetJoinRate)
	 * according to confidence at each iteration
	 * 
	 * @throws Exception
	 */
	public void runStandard() throws Exception {
			
		this.amountToJoin = this.unlabeledSet.getInstances().size() / this.unlabeledSetJoinRate;
		
		int i = 1;
		while (true) {
			generateIterationInfo(i);
			addIterationInfoToHistory();
			
			trainMainCLassifierOverLabeledSet();
			classifyInstancesStandard(this.unlabeledSet);

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
	
	/**
	 * Join only the n best instances that get the 
	 * best confidence of running iteration
	 * 
	 * @throws Exception
	 */
	public void runStandardLazy() throws Exception {
		
		int i = 1;
		while (true) {
			generateIterationInfo(i);
			addIterationInfoToHistory();
			
			trainMainCLassifierOverLabeledSet();
			classifyInstancesStandardLazy(this.unlabeledSet);

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
	
	/**
	 * Join the p percent best instances (unlabeledSetJoinRate)
	 * according to confidence at each iteration but with weight from distance
	 * 
	 * @throws Exception
	 */
	public void runStandardDistanceFactor() throws Exception {
		this.amountToJoin = this.unlabeledSet.getInstances().size() / this.unlabeledSetJoinRate;
		
		int i = 1;
		while (true) {
			generateIterationInfo(i);
			addIterationInfoToHistory();
			
			trainMainCLassifierOverLabeledSet();
			classifyInstancesStandardDistanceFactor(this.unlabeledSet);

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
	
	private void classifyInstancesStandard(Dataset dataset) throws Exception {

		this.missClassifiedInstances = 0;
		
		ArrayList<InstanceResultStandard> standardResults = new ArrayList<InstanceResultStandard>();
		int amount = this.amountToJoin;

		StringBuilder sb = new StringBuilder();
		sb.append("UNLABELED SET ITERATION RESULT: \n\n");
		
		InstanceResultStandard instanceResultStandard;
		
		Iterator<Instance> iterator = this.unlabeledSet.getInstances().iterator();
		while(iterator.hasNext()) {
			Instance instance = iterator.next();
			instanceResultStandard = new InstanceResultStandard(instance);
			instanceResultStandard.addConfidences(this.mainClassifier.distributionForInstance(instance));
			standardResults.add(instanceResultStandard);

			sb.append(instanceResultStandard.outputDataToCsv() + "\n");
		}
		
		Collections.sort(standardResults, InstanceResultStandard.bestConfidenceComparatorDesc);
		
		if(this.unlabeledSet.getInstances().size() < amount*2) {
			amount = this.unlabeledSet.getInstances().size();
		}
		
		for(int i = 0; i < amount; i++) {
			DenseInstance d = (DenseInstance) standardResults.get(i).getInstance().copy();
			d.setClassValue(standardResults.get(i).getBestClass());
			tempSet.addInstance(d); //CAUTION
			
			if(standardResults.get(i).getInstance().classValue() != standardResults.get(i).getBestClass()) {
				this.missClassifiedInstances += 1;
			}
			
			unlabeledSet.getInstances().remove(standardResults.get(i).getInstance());
		}

		this.goodClassifiedInstances = tempSet.getInstances().size();
		sb.append("\n");
		addToHistory(sb.toString());
	}
	
	private void classifyInstancesStandardLazy(Dataset dataset) throws Exception {
		
		ArrayList<InstanceResultStandard> standardResults = new ArrayList<InstanceResultStandard>();
		Double bestConfidence = 0.0;
		
		StringBuilder sb = new StringBuilder();
		sb.append("UNLABELED SET ITERATION RESULT: \n\n");
		
		InstanceResultStandard instanceResultStandard;
		
		Iterator<Instance> iterator = this.unlabeledSet.getInstances().iterator();
		while(iterator.hasNext()) {
			Instance instance = iterator.next();
			instanceResultStandard = new InstanceResultStandard(instance);
			instanceResultStandard.addConfidences(this.mainClassifier.distributionForInstance(instance));
			standardResults.add(instanceResultStandard);
			
			if(instanceResultStandard.getBestConfidence() > bestConfidence) {
				bestConfidence = instanceResultStandard.getBestConfidence();
			}

			sb.append(instanceResultStandard.outputDataToCsv() + "\n");
		}
		
		for(InstanceResultStandard irs: standardResults) {
			if(irs.getBestConfidence() >= bestConfidence) {
				DenseInstance d = (DenseInstance) irs.getInstance().copy();
				d.setClassValue(irs.getBestClass());
				tempSet.addInstance(d); //CAUTION
				unlabeledSet.getInstances().remove(irs.getInstance());
			}
		}
		
		this.goodClassifiedInstances = tempSet.getInstances().size();
		sb.append("\n");
		addToHistory(sb.toString());
	}
	
	
	private void classifyInstancesStandardDistanceFactor(Dataset dataset) throws Exception {
		
		ArrayList<InstanceResultStandard> standardResults = new ArrayList<InstanceResultStandard>();
		Instance[] centroids =  Mathematics.centroidsOf(this.labeledSet.getInstances());
		
		int amount = this.amountToJoin;

		StringBuilder sb = new StringBuilder();
		sb.append("UNLABELED SET ITERATION RESULT: \n\n");
		
		InstanceResultStandard instanceResultStandard;
		
		Iterator<Instance> iterator = this.unlabeledSet.getInstances().iterator();
		while(iterator.hasNext()) {
			Instance instance = iterator.next();
			instanceResultStandard = new InstanceResultStandard(instance);
			instanceResultStandard.addConfidences(this.mainClassifier.distributionForInstance(instance));
			
			Double distance = Mathematics.euclidianDistance(instance, centroids[instanceResultStandard.getBestClassIndex()]);
			instanceResultStandard.setFactor(instanceResultStandard.getBestConfidence() * (1 / distance));
			
			standardResults.add(instanceResultStandard);
		}
		
		Collections.sort(standardResults, InstanceResultStandard.factorComparatorDesc);
		
		for(InstanceResultStandard irs: standardResults) {
			sb.append(irs.outputDataToCsvWithDistanceFactor() + "\n");
		}
		
		if(this.unlabeledSet.getInstances().size() < amount*2) {
			amount = this.unlabeledSet.getInstances().size();
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
	
}
