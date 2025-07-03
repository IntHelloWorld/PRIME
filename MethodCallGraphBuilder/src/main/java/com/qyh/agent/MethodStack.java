package com.qyh.agent;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Stack;

import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultDirectedWeightedGraph;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.jgrapht.nio.gml.GmlExporter;
import org.jgrapht.nio.gml.GmlExporter.Parameter;

public class MethodStack {

    private static ThreadLocal<Stack<String>> threadStack = ThreadLocal.withInitial(() -> new Stack<>());
    private static Graph<String, DefaultWeightedEdge> graph = new DefaultDirectedWeightedGraph<>(DefaultWeightedEdge.class);

    static {
        Runtime.getRuntime().addShutdownHook(new Thread() {
            public void run() {
                GmlExporter<String, DefaultWeightedEdge> exporter = new GmlExporter<>();
                exporter.setParameter(Parameter.EXPORT_EDGE_WEIGHTS, true);
                exporter.setParameter(Parameter.EXPORT_VERTEX_LABELS, true);
                File file = new File("callgraph.graphml");
                try (FileWriter writer = new FileWriter(file)) {
                    exporter.exportGraph(graph, writer);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    public static void push(String content) throws IOException {
        Stack<String> stack = threadStack.get();
        
        // add node
        graph.addVertex(content);

        // add edge
        if (!stack.isEmpty()) {
            DefaultWeightedEdge edge = graph.addEdge(stack.peek(), content);
            if (edge == null) {  // edge already exists
                DefaultWeightedEdge e = graph.getEdge(stack.peek(), content);
                graph.setEdgeWeight(e, graph.getEdgeWeight(e) + 1.0);
            } else {
                graph.setEdgeWeight(edge, 1.0);
            }
        }

        stack.push(content);
    }

    public static void pop() throws IOException {
        Stack<String> stack = threadStack.get();
        if (!stack.isEmpty()) {
            stack.pop();
        }
    }
}