package com.qyh.agent;

import java.lang.instrument.Instrumentation;


public class ClassAgent {

    public static void premain(String argsString, Instrumentation inst) {
        String srcClassPath = null;
        String testClassPath = null;
        String[] args = argsString.split(",");
        for (String arg : args) {
            String[] kv = arg.split("=");
            String key = kv[0];
            String value = kv[1];
            switch (key) {
                case "srcClassPath":
                    srcClassPath = value;
                    break;
                case "testClassPath":
                    testClassPath = value;
                    break;
                default:
                    System.out.println("unknown arg: " + key);
                    return;
            }
        }

        if (srcClassPath == null || testClassPath == null) {
            System.err.println("srcClassPath or testClassPath is null");
            return;
        }

        ClassTransformer transformer = new ClassTransformer(srcClassPath, testClassPath);
        inst.addTransformer(transformer);
    }
}