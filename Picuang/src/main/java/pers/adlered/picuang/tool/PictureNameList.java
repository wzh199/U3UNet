package pers.adlered.picuang.tool;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;

public class PictureNameList {
    private static LinkedList<String> front, belowRGB, belowBinary;

    static {
        front = new LinkedList<String>();
        belowRGB = new LinkedList<String>();
        belowBinary = new LinkedList<String>();
    }

    public static LinkedList<String> getFront() {
        return front;
    }

    public static LinkedList<String> getBelowRGB() {
        return belowRGB;
    }

    public static LinkedList<String> getBelowBinary() {
        return belowBinary;
    }
}
