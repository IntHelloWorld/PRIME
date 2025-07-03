package com.qyh.agent;

import java.lang.instrument.ClassFileTransformer;
import java.lang.instrument.IllegalClassFormatException;
import java.lang.String;
import java.security.ProtectionDomain;
import java.util.HashSet;
import javassist.CannotCompileException;
import javassist.ClassPool;
import javassist.CtBehavior;
import javassist.CtClass;
import javassist.CtConstructor;
import javassist.CtMethod;
import javassist.NotFoundException;
import javassist.bytecode.CodeAttribute;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class ClassTransformer implements ClassFileTransformer {
    static FileWriter loadClassesFileWriter;
    private String srcClassPath;
    private String testClassPath;

    private final static HashSet<String> outerClasses = new HashSet<String>();

    static {
        Runtime.getRuntime().addShutdownHook(new Thread() {
            public void run() {
                try {
                    loadClassesFileWriter.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });
        File loadClassesFile = new File("loaded_classes.txt");
        try {
            loadClassesFileWriter = new FileWriter(loadClassesFile);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public ClassTransformer(String srcClassPath, String testClassPath) {
        this.srcClassPath = srcClassPath;
        this.testClassPath = testClassPath;
    }

    @Override
    public byte[] transform(ClassLoader loader, String className, Class<?> classBeingRedefined,
            ProtectionDomain protectionDomain, byte[] classfileBuffer) throws IllegalClassFormatException {

        // only transform classes in target path
        String classPath = protectionDomain.getCodeSource().getLocation().getPath();
        if (!classPath.contains(srcClassPath) && !classPath.contains(testClassPath)) {
            return null;
        }

        // do not transform MethodStack
        if (className.contains("com/qyh/agent/MethodStack")) {
            return null;
        }

        className = className.replace("/", ".");
        try {
            processClassName(className);

            ClassPool pool = ClassPool.getDefault();
            CtClass ctClass = pool.get(className);
            CtMethod[] cms = ctClass.getDeclaredMethods();
            for (CtMethod cm : cms) {
                transformMethod(cm);
            }
            CtConstructor[] ccs = ctClass.getConstructors();
            for (CtConstructor cc : ccs) {
                transformMethod(cc);
            }
            return ctClass.toBytecode();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private void processClassName(String className) {
        // solve className, consider inner class
        String outerClassName = className.split("\\$")[0];
        if (!outerClasses.contains(outerClassName)) {
            outerClasses.add(outerClassName);
            try {
                loadClassesFileWriter.write(className + "\n");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }


    private void transformMethod(CtBehavior method)
            throws CannotCompileException, NotFoundException {

        CodeAttribute codeAttribute = method.getMethodInfo().getCodeAttribute();
        // if abstract or native method, codeAttribute will be null
        if (codeAttribute == null) {
            return;
        }
        // Get the method's line number range in source code
        int startLine = method.getMethodInfo().getLineNumber(0);
        int endLine = method.getMethodInfo().getLineNumber(codeAttribute.getCodeLength());
        String lineRange = "(" + startLine + "-" + endLine + ")";

        CtClass declaringClass = method.getDeclaringClass();
        String pkgName = declaringClass.getPackageName();
        String className = declaringClass.getSimpleName();
        String content = pkgName + "@" + className + ":" + method.getName() + lineRange;
        
        method.insertBefore(
            "com.qyh.agent.MethodStack.push(\"" + content + "\");"
        );
        method.insertAfter(
            "com.qyh.agent.MethodStack.pop();"
        );
    }
}
