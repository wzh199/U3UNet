package pers.adlered.picuang.controller.websocket;

import org.springframework.stereotype.Component;
import pers.adlered.picuang.tool.PictureNameList;
import pers.adlered.picuang.tool.ToolBox;

import java.util.Base64;
import java.util.Base64.Decoder;

import javax.websocket.*;
import javax.websocket.server.PathParam;
import javax.websocket.server.ServerEndpoint;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;


/**
 * @Author: 杨捷宁
 * @DateTime: 2022/5/7 14:10
 * @Description: 该类用于 提供websocket服务
 */
@ServerEndpoint(value = "/webSocketFireAreaImage/{sid}/{type}")
@Component

public class WebSocketFireAreaImage {
    /**
     * 静态变量，用来记录当前在线连接数,线程安全
     */
    private static AtomicInteger onlineNum = new AtomicInteger();

    /**
     * concurrent包的线程安全Set，用来存放每个客户端对应的WebSocketServer对象。
     */
    private static ConcurrentHashMap<String, Session> SessionPools = new ConcurrentHashMap<>();


    /**
     * 向对应客户端发送信息
     *
     * @param session
     * @param message
     * @author 杨捷宁
     * @date 2022/5/7 19:29
     */
    public static void sendMessage(Session session, String message) throws IOException {
        if (session != null) {
            synchronized (session) {
                session.getBasicRemote().sendText(message);
            }
        }
    }

    public static void tryToSend(double totalPrice) {
        if (SessionPools.size() == 0) {
            return;
        }

        for (Session session : SessionPools.values()) {
            try {
                sendMessage(session, "");
            } catch (Exception e) {
                e.printStackTrace();
                continue;
            }
        }
    }

    /**
     * 建立连接成功调用
     *
     * @param session
     * @param userName 用户名称,用于记录和删除session
     * @param type     需求类型
     * @author 杨捷宁
     * @date 2022/5/7 19:30
     */
    @OnOpen
    public void onOpen(Session session, @PathParam(value = "sid") String userName, @PathParam(value = "type") String type) {

        SessionPools.put(userName, session);

        addOnlineCount();
        System.out.println(userName + "加入webSocketFireAreaImage！当前人数为" + onlineNum);
        try {
//            sendMessage(session, "欢迎" + userName + "加入连接！");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 关闭连接时调用
     *
     * @author 杨捷宁
     * @date 2022/5/7 9:45
     */
    @OnClose
    public void onClose(@PathParam(value = "sid") String userName) {

        SessionPools.remove(userName);

        subOnlineCount();
    }

    //收到客户端信息
    @OnMessage
    public void onMessage(String data, Session session, @PathParam(value = "sid") String userName, @PathParam(value = "type") String type) throws IOException {
//        Logger.log("Received video data size:" + data);
//        Decoder decoder = Base64.getDecoder();
        try {
//            // Base64解码
//            byte[] bytes = decoder.decode(data.substring(2, data.length() - 1));
//            for (int i = 0; i < bytes.length; ++i) {
//                if (bytes[i] < 0) {// 调整异常数据
//                    bytes[i] += 256;
//                }
//            }
//            String[] time = ToolBox.getDirByTime();
//            String path = ToolBox.getPicStoreDir() + time[1] + type + "/" + time[2];
//            String suffixName = Integer.toString(new Random(1).nextInt(10));
//            String fileName = UUID.randomUUID() + suffixName;
//            // 生成jpeg图片
//            File dest = new File(path + fileName);
//            if (!dest.getParentFile().exists()) {
//                dest.getParentFile().mkdirs();
//            }
//            OutputStream out = new FileOutputStream(path + fileName + ".jpg");
//            out.write(bytes);
//            out.flush();
//            out.close();
//            String filename = dest.getName();
//            String archive_url = "uploadImages/" + time[1] + type + "/" + filename + ".jpg";
//            PictureNameList.getFront().add(archive_url);
//            for (ConcurrentHashMap.Entry<String, Session> ce : SessionPools.entrySet()) {
//                ce.getValue().getBasicRemote().sendText(archive_url);
//            }
            SessionPools.get("vue").getBasicRemote().sendText(data);


            //====

        } catch (Exception e) {
            System.out.println(e);
        }

//        try {
//            session.getBasicRemote().sendBinary(ByteBuffer.wrap(data));
//        } catch (IOException e) {
//            Logger.log("Failed to send data: " + session.getId());
//        }
    }

    //错误时调用
    @OnError
    public void onError(Session session, Throwable throwable) {
        System.out.println("发生错误");
        throwable.printStackTrace();
    }

    public static void addOnlineCount() {
        onlineNum.incrementAndGet();
    }

    public static void subOnlineCount() {
        onlineNum.decrementAndGet();
    }

}
