package br.ufrn.imd.ebssb.core;

import java.util.ArrayList;

import br.ufrn.imd.ebssb.results.FoldResult;
import br.ufrn.imd.ebssb.results.InstanceResult;
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

	protected Dataset labeledSet;
	protected Dataset unlabeledSet;

	protected Dataset tempSet;

	protected Classifier classifier;

	private ArrayList<Classifier> bc; // boost committee

	protected ArrayList<Classifier> pool;
	protected double agreementThreshold = 75;

	protected int labeledSetPercentual = 10;
	protected int goodClassifiedInstances = 0;
	protected int missClassifiedInstances = 0;
	private Double initialWeight = 1.0;

	protected FoldResult result;
	protected String history;
	protected String iterationInfo;

	public EbSsBoost(Dataset testSet, Dataset validationSet) {

		this.result = new FoldResult();
		this.history = new String();

		this.bc = new ArrayList<Classifier>();

		this.validationSet = new Dataset(validationSet);
		this.testSet = new Dataset(testSet);

		this.pool = new ArrayList<Classifier>();
		populatePool();
	}

	public void initWeights() {
		this.testSet.initInstancesWeight(initialWeight);
	}

	public void runEbSsBoost() throws Exception {
		buildTestSetByStratifiedSplit();
		initWeights();
		trainClassifiersPool();
		classifyUnlabelledByPool();
		
		//parei aqui
		
	}

	/**
	 * Build test set following proportions between labelled and unlabelled
	 * instances within testSet. After the built, the unlabelled myInstances will
	 * have -1.0 for the instanceClass. On the other hand, the labelled myInstances
	 * will have the instanceClass value equal to the instance inside
	 * testSet.instances
	 * 
	 */
	protected void buildTestSetByStratifiedSplit() {
		testSet.getInstances().stratify(10);

		Instances labelled = testSet.getInstances().testCV(10, 0);
		Instances unlabelled = testSet.getInstances().trainCV(10, 0);

		this.labeledSet = new Dataset(labelled);

		this.testSet.clearInstances();

		for (Instance i : labelled) {
			this.testSet.addLabelledInstance(i);
		}

		for (Instance i : unlabelled) {
			this.testSet.addInstance(i);
		}
	}

	public void trainClassifiersPool() throws Exception {
		for (Classifier c : pool) {
			c.buildClassifier(this.labeledSet.getInstances());
		}
	}

	public void classifyUnlabelledByPool() throws Exception {

		for (MyInstance m : this.testSet.getMyInstances()) {
			if (m.getInstanceClass() == -1) {
				m.addInstanceResult(new InstanceResult(m.getInstance()));
				for (Classifier c : this.pool) {
					m.getResults().get(m.getResults().size() - 1).addPrediction(c.classifyInstance(m.getInstance()));
				}
			}
		}

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
	}

}
