package org.lenskit.mf.hmmsvd;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Test;
import org.lenskit.mf.svdfeature.SVDFeatureInstance;

import java.io.*;
import java.util.ArrayList;

/**
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class HmmSVDFeatureModelInferenceTest {

    @Test
    public void inferenceTest() throws IOException, ClassNotFoundException {
        //String modelFile = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/explore11-withlab-clkrat-feas.hmm.tr.exact.hmm";
        //String modelFile = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/hmm11-withlab-clkrat.est.exact.model";
        String modelFile = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/hmm11-withlab-clkrat.est.merge.model";
        String testFile = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/hmmsvd11-withlab-clkrat.tr";
        //String predFile = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/svdfea11-withlab-clkrat.tr";
        //String predFile = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/svdfea11-withlab-clkrat.est.exact.hmm.tr";
        String predFile = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/svdfea11-withlab-clkrat.est.merge.hmm.tr";


        ObjectInputStream fin = new ObjectInputStream(new FileInputStream(modelFile));
        HmmModel model = (HmmModel)(fin.readObject());

        //HmmModel model = new HmmModel(24);
        //model.loadFromTextFile(new File(modelFile));

        HmmSVDFeatureInstanceDAO teDao = new HmmSVDFeatureInstanceDAO(new File(testFile), " ");
        BufferedWriter fout = new BufferedWriter(new FileWriter(predFile));
        HmmSVDFeatureInstance ins = teDao.getNextInstance();

        RealVector sumVec = MatrixUtils.createRealVector(new double[ins.numPos]);
        RealVector labels = MatrixUtils.createRealVector(new double[ins.numPos]);
        while(ins != null) {
            ArrayList<RealVector> gamma = new ArrayList<>(ins.numObs);
            ArrayList<ArrayList<RealVector>> xi = new ArrayList<>(ins.numObs - 1);
            ArrayList<RealVector> probX = new ArrayList<>(ins.numObs);
            model.inference(ins, gamma, xi, probX);
            sumVec.set(0.0);
            labels.set(0.0);
            for (int i=0; i<ins.numObs; i++) {
                sumVec.combineToSelf(1.0, 1.0, gamma.get(i));
                int act = ins.obs.get(i);
                if (act != ins.numPos) {
                    labels.setEntry(act, 1.0);
                }
            }
            for (int j=0; j<ins.numPos; j++) {
                SVDFeatureInstance svdFeaIns = new SVDFeatureInstance(ins.pos2gfeas.get(j), ins.ufeas,
                                                                      ins.pos2ifeas.get(j));
                svdFeaIns.weight = sumVec.getEntry(j);
                //svdFeaIns.weight = 1.0;

                svdFeaIns.label = labels.getEntry(j);
                fout.write(svdFeaIns.toString() + "\n");
            }
            ins = teDao.getNextInstance();
        }
        fout.close();
    }
}
