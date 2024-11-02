package pers.adlered.picuang.tool;

import org.springframework.util.ClassUtils;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;

/**
 * <h3>picuang</h3>
 * <p>工具箱</p>
 *
 * @author : https://github.com/AdlerED
 * @date : 2019-11-06 11:09
 **/
public class ToolBox {
    private static final Set<String> suffixSet;

    static {
        suffixSet = new HashSet<>();
        suffixSet.add(".jpeg");
        suffixSet.add(".jpg");
        suffixSet.add(".png");
        suffixSet.add(".gif");
        suffixSet.add(".svg");
        suffixSet.add(".bmp");
        suffixSet.add(".ico");
        suffixSet.add(".tiff");
    }

    public static String getSuffixName(String filename) {
        String suffixName = filename.substring(filename.lastIndexOf("."));
        suffixName = suffixName.toLowerCase();
        return suffixName;
    }

    public static boolean isPic(String suffixName) {
        return suffixSet.contains(suffixName);
    }

    public static String getPicStoreDir() {
        return ClassUtils.getDefaultClassLoader().getResource("").getPath() + "static/uploadImages/";
    }

    public static File generatePicFile(String suffixName, String ymd, String time, String Type) {
        String path = getPicStoreDir() + ymd + Type + "/" + time ;
        String fileName = UUID.randomUUID() + suffixName;
        return new File(path + fileName);
    }

    public static String[] getDirByTime() {
        Date date = new Date();
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy/MM/dd/HH-mm-");
        SimpleDateFormat simpleDateFormat2 = new SimpleDateFormat("yyyy/MM/dd/");
        SimpleDateFormat simpleDateFormat3 = new SimpleDateFormat("HH-mm-ss-SSS-");
        String[] ret = new String[3];
        ret[0] = simpleDateFormat.format(date);
        ret[1] = simpleDateFormat2.format(date);
        ret[2] = simpleDateFormat3.format(date);
        return ret;
    }

    public static String getINIDir() {
        return new File("config.ini").getAbsolutePath();
    }

    public static boolean deleteUploadImages() {
        File file = new File(getPicStoreDir());
        return deleteFile(file);
    }

    private static boolean deleteFile(File file) {
        //判断文件不为null或文件目录存在
        if (file == null || !file.exists()) {
            System.out.println("无图片路径，无需删除");
            return false;
        }
        //获取目录下子文件
        File[] files = file.listFiles();
        //遍历该目录下的文件对象
        for (File f : files) {
            //判断子目录是否存在子目录,如果是文件则删除
            if (f.isDirectory()) {
                //递归删除目录下的文件
                deleteFile(f);
            } else {
                //文件删除
                f.delete();
                //打印文件名
//                System.out.println("文件名：" + f.getName());
            }
        }
        //文件夹删除
        file.delete();
//        System.out.println("目录名：" + file.getName());
        return true;
    }
}
