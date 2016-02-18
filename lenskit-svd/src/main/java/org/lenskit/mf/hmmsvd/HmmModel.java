package org.lenskit.mf.hmmsvd;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.function.Log;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.StatUtils;
import org.lenskit.solver.objective.LearningInstance;
import org.lenskit.solver.objective.LearningModel;
import org.lenskit.solver.objective.RandomInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;

/**
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class HmmModel extends LearningModel {
    private int numPos; //also represents no action
    private RealVector start;
    private RealMatrix trans;
    private RealMatrix emit;
    private double startObj;
    private double transObj;
    private double obsObj;
    private transient HmmSVDFeatureInstanceDAO dao;
    private transient ArrayList<RealVector> gamma;
    private transient ArrayList<ArrayList<RealVector>> xi;
    private transient RealVector startUpdate;
    private transient ArrayList<RealVector> transUpdate;
    private transient ArrayList<RealVector> emitUpdate;
    private transient static Logger logger = LoggerFactory.getLogger(HmmModel.class);

    public HmmModel(int inNumPos) throws IOException {
        numPos = inNumPos;
        startObj = 0.0;
        transObj = 0.0;
        obsObj = 0.0;
    }

    public void setInstanceDAO(HmmSVDFeatureInstanceDAO inDao) {
        dao = inDao;
    }

    public void assignUpdateVariables() {
        startUpdate = MatrixUtils.createRealVector(new double[numPos]);
        transUpdate = new ArrayList<>(numPos);
        emitUpdate = new ArrayList<>(numPos);
        for (int i=0; i<numPos; i++) {
            transUpdate.add(MatrixUtils.createRealVector(new double[numPos]));
            emitUpdate.add(MatrixUtils.createRealVector(new double[numPos + 1]));
        }
    }

    public void loadFromTextFile(File modelFile) {
        try {
            double small = 10e-6;
            BufferedReader reader = new BufferedReader(new FileReader(modelFile));
            String line = reader.readLine();
            numPos = Integer.parseInt(line);
            start = MatrixUtils.createRealVector(new double[numPos]);
            line = reader.readLine();
            String[] fields = line.split(" ");
            for (int i=0; i<numPos; i++) {
                double val = Double.parseDouble(fields[i]);
                if (val == 0.0) {
                    val = small;
                }
                start.setEntry(i, val);
            }
            trans = MatrixUtils.createRealMatrix(numPos, numPos);
            for (int i=0; i<numPos; i++) {
                line = reader.readLine();
                fields = line.split(" ");
                for (int j=0; j<numPos; j++) {
                    double val = Double.parseDouble(fields[j]);
                    if (val == 0.0) {
                        val = small;
                    }
                    trans.setEntry(i, j, val);
                }
            }
            emit = MatrixUtils.createRealMatrix(numPos, numPos + 1);
            for (int i=0; i<numPos; i++) {
                line = reader.readLine();
                fields = line.split(" ");
                for (int j=0; j<numPos+1; j++) {
                    double val = Double.parseDouble(fields[j]);
                    if (val == 0.0) {
                        val = small;
                    }
                    emit.setEntry(i, j, val);
                }
            }
            reader.close();
            assignUpdateVariables();
            logger.debug("{}", start);
            logger.debug("{}", trans);
            logger.debug("{}", emit);
        } catch (IOException e) {}
    }

    public void assignVariables() {
        RandomInitializer randInit = new RandomInitializer();
        start = MatrixUtils.createRealVector(new double[numPos]);
        trans = MatrixUtils.createRealMatrix(numPos, numPos);
        emit = MatrixUtils.createRealMatrix(numPos, numPos + 1);
        randInit.randInitVector(start, true);
        randInit.randInitMatrix(trans, true);
        randInit.randInitMatrix(emit, true);
        assignUpdateVariables();
        logger.debug("{}", start);
        logger.debug("{}", trans);
        logger.debug("{}", emit);
    }

    public void inference(HmmSVDFeatureInstance ins, ArrayList<RealVector> outGamma,
            ArrayList<ArrayList<RealVector>> outXi, ArrayList<RealVector> probX) {
        //compute p(x|z)
        for (int i=0; i<ins.numObs; i++) {
            int act = ins.obs.get(i);
            probX.add(emit.getColumnVector(act));
        }
        //initialize alpha and beta n-1
        ArrayList<RealVector> alphaHat = new ArrayList<>(ins.numObs);
        ArrayList<RealVector> betaHat = new ArrayList<>(ins.numObs);
        RealVector c = MatrixUtils.createRealVector(new double[ins.numObs]);
        //compute alphaHat 0 to n-1
        for (int i=0; i<ins.numObs; i++) {
            RealVector probx = probX.get(i);
            RealVector alpha = null;
            if (i == 0) {
                alpha = probx.ebeMultiply(start);
            } else if (i > 0) {
                alpha = probx.ebeMultiply(trans.preMultiply(alphaHat.get(i - 1)));
            }
            c.setEntry(i, StatUtils.sum(((ArrayRealVector)alpha).getDataRef()));
            alpha.mapDivideToSelf(c.getEntry(i));
            alphaHat.add(alpha);
            betaHat.add(MatrixUtils.createRealVector(new double[numPos]));
        }
        //compute betaHat n-1 to 0
        betaHat.get(ins.numObs - 1).set(1.0);
        for (int j=ins.numObs - 2; j>=0; j--) {
            RealVector betaj = betaHat.get(j);
            RealVector probx = probX.get(j + 1);
            double cj = c.getEntry(j + 1);
            betaj.setSubVector(0, trans.operate(probx.ebeMultiply(betaHat.get(j + 1))).mapDivideToSelf(cj));
        }
        if (Double.isNaN(alphaHat.get(ins.numObs - 1).getEntry(0)) || Double.isNaN(betaHat.get(0).getEntry(0))) {
            int x = 1;
        }
        //compute gamma and xi
        for (int i=0; i<ins.numObs; i++) {
            outGamma.add(alphaHat.get(i).ebeMultiply(betaHat.get(i)));
            if (i > 0) {
                RealVector probx = probX.get(i);
                RealVector betai = betaHat.get(i);
                RealVector alphai = alphaHat.get(i - 1);
                double ci = c.getEntry(i);
                ArrayList<RealVector> subXi = new ArrayList<>(numPos);
                for (int j=0; j<numPos; j++) {
                    subXi.add(probx.ebeMultiply(trans.getRowVector(j))
                            .ebeMultiply(betai).mapMultiplyToSelf(alphai.getEntry(j) * ci));
                }
                outXi.add(subXi);
            }
        }
        int x = 1;
    }

    public double expectation(LearningInstance inIns) {
        HmmSVDFeatureInstance ins;
        if (inIns instanceof HmmSVDFeatureInstance) {
            ins = (HmmSVDFeatureInstance)inIns;
        } else {
            return 0.0;
        }
        //make sure ins.numPos == this.numPos
        gamma = new ArrayList<>(ins.numObs);
        xi = new ArrayList<>(ins.numObs - 1);
        ArrayList<RealVector> probX = new ArrayList<>(ins.numObs);
        inference(ins, gamma, xi, probX);

        //prepare for maximization step
        RealVector gamma0 = gamma.get(0);
        startUpdate.combineToSelf(1.0, 1.0, gamma0);
        for (int i=0; i<numPos; i++) {
            for (int j=0; j<ins.numObs-1; j++) {
                ArrayList<RealVector> cxj = xi.get(j);
                transUpdate.get(i).combineToSelf(1.0, 1.0, cxj.get(i));
            }
            for (int j=0; j<ins.numObs; j++) {
                RealVector gammaj = gamma.get(j);
                int act = ins.obs.get(j);
                emitUpdate.get(i).addToEntry(act, gammaj.getEntry(i));
            }
        }
        //compute the objective value after expectation
        UnivariateFunction log = new Log();
        double startObjVal = gamma0.dotProduct(start.map(log));
        double transObjVal = 0.0;
        for (int i=0; i<ins.numObs-1; i++) {
            ArrayList<RealVector> cxi = xi.get(i);
            for (int j=0; j<numPos; j++) {
                transObjVal += cxi.get(j).dotProduct(trans.getRowVector(j).mapToSelf(log));
            }
        }
        double obsObjVal = 0.0;
        for (int i=0; i<ins.numObs; i++) {
            RealVector gammai = gamma.get(i);
            RealVector probx = probX.get(i);
            for (int j=0; j<numPos; j++) {
                double g = gammai.getEntry(j);
                double p = probx.getEntry(j);
                obsObjVal += (g * Math.log(p));
            }
        }
        startObj -= startObjVal;
        transObj -= transObjVal;
        obsObj -= obsObjVal;
        return -(startObjVal + transObjVal + obsObjVal);
    }

    public LearningModel maximization() { //closed form maximization
        logger.debug("Start: {}, Trans: {}, Obs: {}", startObj, transObj, obsObj);
        startObj = 0.0;
        transObj = 0.0;
        obsObj = 0.0;
        start.setSubVector(0, startUpdate);
        double sum = StatUtils.sum(((ArrayRealVector)start).getDataRef());
        start.mapDivideToSelf(sum);
        for (int i=0; i<numPos; i++) {
            RealVector trUp = transUpdate.get(i);
            sum = StatUtils.sum(((ArrayRealVector)trUp).getDataRef());
            trans.setRowVector(i, trUp.mapDivideToSelf(sum));
            RealVector emUp = emitUpdate.get(i);
            sum = StatUtils.sum(((ArrayRealVector)emUp).getDataRef());
            emit.setRowVector(i, emUp.mapDivideToSelf(sum));
        }
        return null;
    }

    public HmmSVDFeatureInstance getLearningInstance() {
        try {
            HmmSVDFeatureInstance ins = dao.getNextInstance();
            return ins;
        } catch (IOException e) {
            return null;
        }
    }

    public void startNewIteration() {
        startUpdate.set(0.0);
        for (int i=0; i<numPos; i++) {
            transUpdate.get(i).set(0.0);
            emitUpdate.get(i).set(0.0);
        }
        try {
            dao.goBackToBeginning();
        } catch (IOException e) { }
    }
}
