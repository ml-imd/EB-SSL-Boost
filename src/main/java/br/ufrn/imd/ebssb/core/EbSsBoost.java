package br.ufrn.imd.ebssb.core;

import java.util.ArrayList;
import java.util.Iterator;

import br.ufrn.imd.ebssb.metrics.Measures;
import br.ufrn.imd.ebssb.metrics.Prediction;
import br.ufrn.imd.ebssb.results.FoldResult;
import br.ufrn.imd.ebssb.results.InstanceResult;
import br.ufrn.imd.ebssb.results.IterationInfo;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

public class EbSsBoost {

	private Dataset validationSet;
	private Dataset testSet;

	private Dataset labelledSet;

	private Dataset boostSubSet;
	private Dataset tempSet;

	private ArrayList<ClassifierWithInfo> bc; // boost committee
	private int bcSize = 10;

	private ArrayList<Classifier> pool;
	private double agreementThreshold = 75; // agreement percent
	private int agreementValue; // number of votes - target

	private Double initialWeight = 1.0;
	private Double weightRate;

	private int boostSubsetPercent = 20;
	private int boostSubsetAmount;

	private MyRandom random;

	private FoldResult foldResult;
	private IterationInfo iterationInfo;
	
	private boolean usingClassifierWeight = true;

	public EbSsBoost(Dataset testSet, Dataset validationSet, int seed) {

		this.foldResult = new FoldResult();
		this.random = new MyRandom(seed);

		this.bc = new ArrayList<ClassifierWithInfo>();

		this.validationSet = new Dataset(validationSet);
		this.testSet = new Dataset(testSet);
		this.tempSet = new Dataset(testSet);
		this.tempSet.clearInstances();

		this.pool = new ArrayList<Classifier>();
		populatePool();

		buildTestSetByStratifiedSplit();
		this.boostSubsetAmount = testSet.getInstances().size() * this.boostSubsetPercent / 100;

		computeWeightRate();
		initWeights();
	}

	public void runEbSsBoost() throws Exception {
		System.out.print(" Boost iteration: ");
		while (this.bc.size() < this.bcSize) {
			System.out.print((bc.size()+1) + " ");
			this.iterationInfo = new IterationInfo();

			trainClassifiersPool();
			classifyUnlabelledByPool();

			sampleDataForBoostClassifier();

			pinLabelsInTestUsingPoolPredictions();

			trainBoostClassifierWithBcSubSet();
			renewlabelledAndUnlabelldSets();
			
			updateIntancesWeight();

			testBcOverLabelledInstances();

			this.foldResult.addIterationInfo(iterationInfo);
		}
		testBC();
		System.out.println();
	}

	private void initWeights() {
		this.testSet.initInstancesWeight(initialWeight);
	}

	/**
	 * Build test set following proportions between labelled and unlabelled
	 * instances within testSet. After the built, the unlabelled myInstances will
	 * have -1.0 for the instanceClass. On the other hand, the labelled myInstances
	 * will have the instanceClass value equal to the instance inside
	 * testSet.instances
	 * 
	 */
	private void buildTestSetByStratifiedSplit() {
		testSet.getInstances().stratify(10);

		Instances labelled = testSet.getInstances().testCV(10, 0);
		Instances unlabelled = testSet.getInstances().trainCV(10, 0);

		this.labelledSet = new Dataset(labelled);

		this.testSet.clearInstances();

		for (Instance i : labelled) {
			this.testSet.addLabelledInstance(i);
		}

		for (Instance i : unlabelled) {
			this.testSet.addInstance(i);
		}

		this.testSet.storePositions();

		// Log info
		this.storeSizes(labelled.size(), unlabelled.size(), this.validationSet.getInstances().size());
		
	}
	
	private void trainClassifiersPool() throws Exception {
		for (Classifier c : pool) {
			c.buildClassifier(this.labelledSet.getInstances());
		}
	}

	private void classifyUnlabelledByPool() throws Exception {
		InstanceResult result;
		Iterator<MyInstance> iterator = this.testSet.getMyInstances().iterator();

		while (iterator.hasNext()) {
			MyInstance m = iterator.next();
			if (m.getInstanceClass() == -1) {
				result = new InstanceResult(m.getInstance());

				for (Classifier c : this.pool) {
					result.addPrediction(c.classifyInstance(m.getInstance()));
				}
				m.setResult(result);
			}
		}
	}

	private void sampleDataForBoostClassifier() {

		this.boostSubSet = new Dataset(tempSet.getInstances());
		this.boostSubSet.clearInstances();

		// building the tempSet for performing the weighted draw over it
		for (MyInstance m : this.testSet.getMyInstances()) {
			if (m.getInstanceClass() != -1.0 || m.getResult().getBestAgreement() >= this.agreementValue) {
				MyInstance mNew = new MyInstance(m);
				this.tempSet.addMyInstance(mNew);
			}
		}

		// sampling
		while (this.boostSubSet.getMyInstances().size() < this.boostSubsetAmount) {
			MyInstance m = new MyInstance(this.tempSet.drawOne(random));
			this.boostSubSet.addMyInstance(m);
		}

		Iterator<MyInstance> iterator = this.boostSubSet.getMyInstances().iterator();
		// pin label inside instance within boostSubSet
		while (iterator.hasNext()) {
			MyInstance m = iterator.next();
			// if the sampled instance is unlabelled, set label from pool in instance
			if (m.getInstanceClass() == -1) {
				// pin class in boostSubSet intance
				m.getInstance().setClassValue(m.getResult().getBestClass());
				m.setInstanceClass(m.getResult().getBestClass());
			}
		}
		this.tempSet.clearInstances();

		// Log info
		this.iterationInfo.setBoostSubSetSize(this.boostSubSet.getMyInstances().size());
	}

	/**
	 * this methods looks to the current boostSubSet and pins the label defined by the
	 * pool of classifiers.
	 * 
	 * In other words, if some instance sampled for current boostSubSet is unlabelled, then this
	 * same label is pinned in testset instances using the class defined by the pool.
	 */
	private void pinLabelsInTestUsingPoolPredictions() {
		int c = 0;
		for (MyInstance m : this.boostSubSet.getMyInstances()) {
			// if instance had not a label
			if (m.getResult() != null) {
				int i = this.testSet.getPositions().get(m.getHashId());
				double classValue = this.testSet.getMyInstances().get(i).getResult().getBestClass();

				this.testSet.getMyInstances().get(i).setInstanceClass(classValue);
				c++;
			}
		}

		// Log info
		this.iterationInfo.setInstancesSampledAndLabelledByPool(c);
	}

	private void trainBoostClassifierWithBcSubSet() {

		// weka.classifiers.trees.J48 -C 0.05 -M 2 (74.4792)

		J48 j48 = new J48();
		try {
			j48.setOptions(weka.core.Utils.splitOptions("-C 0.05 -M 2"));
			j48.buildClassifier(this.boostSubSet.getInstances());
			
			ClassifierWithInfo classifier = new ClassifierWithInfo(j48);
			
			if(usingClassifierWeight) {
				double sumWeights = 0;
				double totalWeights = 0;
				
				for(MyInstance minstance : boostSubSet.getMyInstances()) {
					if (minstance.getInstanceClass() != -1) {
						double output = j48.classifyInstance(minstance.getInstance());
						double weight = minstance.getWeight();
						totalWeights += weight;
						if(minstance.getInstanceClass() != output) {
							sumWeights += weight;
						}
					}
				}
				
				double correction = 0.00001;
				double err = (sumWeights + correction) / (totalWeights + correction);
				int numClasses = boostSubSet.getInstances().numClasses();
				double alpha = Math.log((1.0 - err)/err) + Math.log(numClasses - 1);
				
				classifier.setWeight(alpha);
			}	
				
			this.bc.add(classifier);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void renewlabelledAndUnlabelldSets() {

		this.labelledSet.clearInstances();
		for (MyInstance m : this.testSet.getMyInstances()) {
			if (m.getInstanceClass() != -1) {
				this.labelledSet.addMyInstance(m);
			}
		}
	}
	
	private void updateIntancesWeight() throws Exception {
		Iterator<MyInstance> iterator = this.testSet.getMyInstances().iterator();
		//Version considering only the last classifier from bc
		ClassifierWithInfo classifier = bc.get(bc.size() - 1);
		
		while (iterator.hasNext()) {
			MyInstance m = iterator.next();

			if (m.getInstanceClass() != -1) {
				Double predictedClass = classifier.getClassifier().classifyInstance(m.getInstance());
				
				if(usingClassifierWeight) {
					updateInstanceWeightAccordingToPredictedClassAndAlpha(m, predictedClass, classifier.getWeight());
				} else {
					updateInstanceWeightAccordingToPredictedClass(m, predictedClass);
				}
			}
		}
	}
	
	// Instance weight update - Cephas
	private void updateInstanceWeightAccordingToPredictedClass(MyInstance m, Double predictedClass) {
		if (predictedClass != m.getInstanceClass().doubleValue()) {
			m.increaseWeight(this.weightRate);
			this.testSet.increaseTotalWeight(this.weightRate);
		}
	}
	
	// Instance weight update - algorithm
	private void updateInstanceWeightAccordingToPredictedClassAndAlpha(MyInstance m, Double predictedClass, Double alpha) {
		Double newWeight = m.getWeight();
		
		if (predictedClass != m.getInstanceClass().doubleValue()) {
			newWeight *= Math.exp(alpha);
			Double increase = newWeight - m.getWeight();
			
			m.setWeight(newWeight);
			this.testSet.increaseTotalWeight(increase);
		}
	}

	private void testBcOverLabelledInstances() throws Exception {

		InstanceResult result;
		Iterator<MyInstance> iterator = this.testSet.getMyInstances().iterator();

		int labelledInstances = 0;
		int bcHit = 0;
		int bcWrong = 0;

		while (iterator.hasNext()) {
			MyInstance m = iterator.next();

			// if the instance is labelled
			if (m.getInstanceClass() != -1) {
				labelledInstances++;
				result = new InstanceResult(m.getInstance());

				// test it with the bc
				for (ClassifierWithInfo info : this.bc) {
					Classifier c = info.getClassifier();
					if(usingClassifierWeight) {
						result.addPredictionWithWeight(c.classifyInstance(m.getInstance()), info.getWeight());
					} else {
						result.addPrediction(c.classifyInstance(m.getInstance()));
					}
				}
				m.setBoostEnsembleResult(result);

				// if the bc prediction is different of pinned label -> bc wrong and the weight
				// is augmented
				if (m.getBoostEnsembleResult().getBestClass().doubleValue() != m.getInstanceClass().doubleValue()) {
					bcWrong++;
				}
				// else, the bc predicted correctly and weight of the instance is decreased
				else {
					bcHit++;
				}
			}
		}

		this.iterationInfo.setBoostEnsembleErrors(bcWrong);
		this.iterationInfo.setBoostEnsembleHits(bcHit);
		this.iterationInfo.setLabelledInstances(labelledInstances);
		this.iterationInfo.setUnlabelledInstances(this.testSet.getInstances().size() - labelledInstances);
		this.iterationInfo.setBoostSubSetSize(this.boostSubSet.getInstances().size());

		this.iterationInfo.setBoostSubsetSummary(this.boostSubSet.getMyInstancesSummary());
		this.iterationInfo.setTestSetSummary(this.testSet.getMyInstancesSummary());

	}

	private void storeSizes(int labelledSetSize, int unlabelledSetSize, int validationSetSize) {
		this.foldResult.setLabelledSetSize(labelledSetSize);
		this.foldResult.setUnlabelledSetSize(unlabelledSetSize);
		this.foldResult.setValidationSetSize(validationSetSize);
	}

	private void computeWeightRate() {
		this.weightRate = 1.0;
	}

	private void testBC() throws Exception {
		
		Prediction pred = new Prediction(this.validationSet);
		InstanceResult ir;
		
		for (Instance i : this.validationSet.getInstances()) {
			ir = new InstanceResult(i);
			for (ClassifierWithInfo info : this.bc) {
				Classifier c = info.getClassifier();
				if(usingClassifierWeight){
					ir.addPredictionWithWeight(c.classifyInstance(i), info.getWeight());
				} else {
					ir.addPrediction(c.classifyInstance(i));
				}
			}
			pred.addPrediction(i.classValue(), ir.getBestClass());
		}
		pred.buildMetrics();
		
		Measures measures = new Measures(pred);
		foldResult.setAccuracy(measures.getAccuracy());
		foldResult.setError(measures.getError());
		foldResult.setfMeasure(measures.getFmeasureMean());
		foldResult.setPrecision(measures.getPrecisionMean()); 
		foldResult.setRecall(measures.getRecallMean());
		
	}

	private void populatePool() {
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

		double agreementValue = pool.size() * agreementThreshold / 100;
		this.agreementValue = (int) agreementValue;
	}

	public FoldResult getFoldResult() {
		return foldResult;
	}
	
	
}
