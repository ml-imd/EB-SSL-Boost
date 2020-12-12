package br.ufrn.imd.ebssb.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import br.ufrn.imd.ebssb.results.InstanceResult;
import br.ufrn.imd.ebssb.results.InstanceResultStandard;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;

public class SelfTrainingEnsembleBasedTest extends SelfTrainingEnsembleBased{

	public SelfTrainingEnsembleBasedTest(Dataset testSet, Dataset validationSet) {
		super(testSet, validationSet);
	}
	
	public void runTest() throws Exception {
		
		this.amountToJoin = this.unlabeledSet.getInstances().size() / this.unlabeledSetJoinRate;
		
		int i = 1;
		while (true) {
			generateIterationInfo(i);
			addIterationInfoToHistory();
			
			trainMainCLassifierOverLabeledSet();
			trainClassifiersPool();
			
			classifyInstancesStandardTest(this.unlabeledSet);

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
	
	
	private void classifyInstancesStandardTest(Dataset dataset) throws Exception {

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
			
			InstanceResult instanceResult = new InstanceResult(d);
			for(Classifier c: this.pool) {
				instanceResult.addPrediction(c.classifyInstance(d));
			}
			
			d.setClassValue(instanceResult.getBestClass());
			tempSet.addInstance(d); //CAUTION
			
			if(standardResults.get(i).getInstance().classValue() != instanceResult.getBestClass()) {
				this.missClassifiedInstances += 1;
			}
			
			unlabeledSet.getInstances().remove(standardResults.get(i).getInstance());
		}

		this.goodClassifiedInstances = tempSet.getInstances().size();
		sb.append("\n");
		addToHistory(sb.toString());
	}

		
}
