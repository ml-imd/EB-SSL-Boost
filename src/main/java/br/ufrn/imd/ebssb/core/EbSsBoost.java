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

	protected Dataset validationSet;
	protected Dataset testSet;

	protected Dataset labelledSet;

	protected Dataset boostSubSet;
	protected Dataset tempSet;

	protected Classifier classifier;

	private ArrayList<Classifier> bc; // boost committee
	private int bcSize = 10;

	protected ArrayList<Classifier> pool;
	protected double agreementThreshold = 75; // agreement percent
	protected int agreementValue; // number of votes - target

	protected int labeledSetPercentual = 10;
	private Double initialWeight = 1.0;
	private Double weightRate;

	protected int goodClassifiedInstances = 0;
	protected int missClassifiedInstances = 0;

	protected int boostSubsetPercent = 20;
	protected int boostSubsetAmount;

	protected MyRandom random;

	protected FoldResult result;
	protected String history;
	protected IterationInfo iterationInfo;

	public EbSsBoost(Dataset testSet, Dataset validationSet, int seed) {

		this.result = new FoldResult();
		this.history = new String();
		this.random = new MyRandom(seed);

		this.bc = new ArrayList<Classifier>();

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

		while (this.bc.size() < this.bcSize) {

			this.iterationInfo = new IterationInfo();

			trainClassifiersPool();
			classifyUnlabelledByPool();

			sampleDataForBoostClassifier();

			pinLabelsInTestUsingPoolPredictions();

			trainBoostClassifierWithBcSubSet();
			renewlabelledAndUnlabelldSets();

			testBcOverLabelledInstances();

			this.result.addIterationInfo(iterationInfo);
			// test();
		}
		System.out.println(this.result.foldResultSummarytable());
		testBC();
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
	
	/** 
	 * método precisa ser observado pois o labelled set muda e vai precisar ser
	 * atualizado a cada iteração.
	*/
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
	 * this methods looks to the current boostSubSet and pins the label defined from
	 * pool of classifiers.
	 * 
	 * if some instance sampled for current boostSubSet is unlabelled, then this
	 * same label in test is pinned using class defined from pool.
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
		} catch (Exception e) {
			e.printStackTrace();
		}

		this.bc.add(j48);
	}

	private void renewlabelledAndUnlabelldSets() {

		this.labelledSet.clearInstances();
		for (MyInstance m : this.testSet.getMyInstances()) {
			if (m.getInstanceClass() != -1) {
				this.labelledSet.addMyInstance(m);
			}
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
				for (Classifier c : this.bc) {
					result.addPrediction(c.classifyInstance(m.getInstance()));
				}
				m.setBoostEnsembleResult(result);

				// if the bc prediction is different of pinned label -> bc wrong and the weight
				// is augmented
				if (m.getBoostEnsembleResult().getBestClass().doubleValue() != m.getInstanceClass().doubleValue()) {
					m.increaseWeight(this.weightRate);
					this.testSet.increaseTotalWeight(this.weightRate);
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
		this.result.setLabelledSetSize(labelledSetSize);
		this.result.setUnlabelledSetSize(unlabelledSetSize);
		this.result.setValidationSetSize(validationSetSize);
	}

	private void computeWeightRate() {
		this.weightRate = 1.0;
	}

	private void testBC() throws Exception {
		Prediction pred = new Prediction(this.validationSet);
		InstanceResult ir;
		for (Instance i : this.validationSet.getInstances()) {
			ir = new InstanceResult(i);
			for (Classifier c : this.bc) {
				ir.addPrediction(c.classifyInstance(i));
			}
			pred.addPrediction(i.classValue(), ir.getBestClass());
		}
		pred.buildMetrics();
		
		Measures measures = new Measures(pred);
		
		result.setAccuracy(measures.getAccuracy());
		result.setError(measures.getError());
		result.setfMeasure(measures.getFmeasureMean());
		result.setPrecision(measures.getPrecisionMean()); 
		result.setRecall(measures.getRecallMean());
		
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

}
