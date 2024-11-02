package pers.adlered.picuang.controller.api;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import pers.adlered.picuang.controller.api.bean.PicProp;
import pers.adlered.picuang.log.Logger;
import pers.adlered.picuang.tool.IPUtil;
import pers.adlered.picuang.tool.PictureNameList;
import pers.adlered.picuang.tool.ToolBox;

import javax.servlet.http.HttpServletRequest;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Queue;

/**
 * <h3>picuang</h3>
 * <p>查看历史记录API</p>
 *
 * @author : https://github.com/AdlerED
 * @date : 2019-11-06 16:24
 **/
@Controller
public class History {
    @RequestMapping("/api/list")
    @ResponseBody
    public List<PicProp> list(HttpServletRequest request, String year, String month, String day, String type) {
        List<PicProp> list = new ArrayList<>();
        File file = new File(getHome(request) + year + "/" + month + "/" + day + "/" + type + "/");
        File[] files = listFiles(file);

        try {
            for (File k : files) {
                if (k.isFile()) {
                    String[] splitName = k.getName().split("-");
                    PicProp picProp = new PicProp();
                    picProp.setTime(splitName[0] + ":" + splitName[1]);
                    picProp.setFilename(k.getName());
                    picProp.setPath("/uploadImages/" + year + "/" + month + "/" + day + "/" + type + "/" + k.getName());
                    picProp.setIp(IPUtil.getIpAddr(request));
                    list.add(picProp);
                }
            }
        } catch (NullPointerException NPE) {
            logNpe();

        }
        Collections.reverse(list);
        return list;
    }

    @RequestMapping("/api/getLastPic")
    @ResponseBody
    @CrossOrigin
    public PicProp getLastPic(HttpServletRequest request, String year, String month, String day, String type) {
        String address = "";
        switch (type) {
            case "front":
                address = PictureNameList.getFront().peek();
                if (address != null)
                    PictureNameList.getFront().removeFirst();
                break;
            case "belowRGB":
                address = PictureNameList.getBelowRGB().peek();
                if (address != null)
                    PictureNameList.getBelowRGB().removeFirst();
                break;
            case "belowBinary":
                address = PictureNameList.getBelowBinary().peek();
                if (address != null)
                    PictureNameList.getBelowBinary().removeFirst();
                break;
        }
        if (address == null) {
            return null;
        }
        File file = new File(address);
        System.out.println(address);

        PicProp picProp = new PicProp();
        try {
            if (file.isFile()) {
                String[] splitName = file.getName().split("-");
                picProp.setTime(splitName[0] + ":" + splitName[1]);
                picProp.setFilename(file.getName());
                picProp.setPath("/uploadImages/" + year + "/" + month + "/" + day + "/" + type + "/" + file.getName());
                picProp.setIp(IPUtil.getIpAddr(request));
            }

        } catch (NullPointerException NPE) {
            logNpe();

        }
        return picProp;

    }


    @RequestMapping("/api/day")
    @ResponseBody
    public List<String> day(HttpServletRequest request, String year, String month) {
        File file = new File(getHome(request) + year + "/" + month + "/");
        File[] list = listFiles(file);
        List<String> lists = new ArrayList<>();
        try {
            for (File i : list) {
                if (i.isDirectory()) {
                    lists.add(i.getName());
                }
            }
        } catch (NullPointerException NPE) {
            logNpe();
        }
        Collections.reverse(lists);
        return lists;
    }

    @RequestMapping("/api/month")
    @ResponseBody
    public List<String> month(HttpServletRequest request, String year) {
        StringBuilder sb = new StringBuilder();
        File file = new File(getHome(request) + year + "/");
        File[] list = listFiles(file);
        List<String> lists = new ArrayList<>();
        try {
            for (File i : list) {
                if (i.isDirectory()) {
                    lists.add(i.getName());
                }
            }
        } catch (NullPointerException NPE) {
        }
        Collections.reverse(lists);
        return lists;
    }

    @RequestMapping("/api/year")
    @ResponseBody
    public List<String> year(HttpServletRequest request) {
        File file = new File(getHome(request));
        File[] list = listFiles(file);
        List<String> lists = new ArrayList<>();
        try {
            for (File i : list) {
                if (i.isDirectory()) {
                    lists.add(i.getName());
                }
            }
        } catch (NullPointerException NPE) {
        }
        Collections.reverse(lists);
        return lists;
    }

    private String getHome(HttpServletRequest request) {
//        String addr = IPUtil.getIpAddr(request).replaceAll("\\.", "/").replaceAll(":", "/");
        return ToolBox.getPicStoreDir() + "/";
    }

    private File[] listFiles(File file) {
        File[] files = file.listFiles();
        return files == null ? new File[0] : files;
    }

    private void logNpe() {
        Logger.log(String.format("A null pointer exception occurred in [%s]", this.getClass().getName()));
    }
}
