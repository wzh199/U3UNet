package pers.adlered.picuang.controller.websocket;

import com.alibaba.fastjson.JSONObject;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.socket.server.standard.SpringConfigurator;
import pers.adlered.picuang.log.Logger;
import pers.adlered.picuang.tool.PictureNameList;
import pers.adlered.picuang.tool.ToolBox;

import javax.websocket.*;
import javax.websocket.server.HandshakeRequest;
import javax.websocket.server.PathParam;
import javax.websocket.server.ServerEndpoint;
import javax.websocket.server.ServerEndpointConfig;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;


/**
 * @Author: 杨捷宁
 * @DateTime: 2022/5/7 14:10
 * @Description: 该类用于 提供websocket服务
 */
@ServerEndpoint(value="/UAVHeightWebSocket/{sid}/{type}")
@Component

public class UAVHeightWebSocketServer { //名字起错了实际上是无人机【状态】socket
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
     * @author 杨捷宁
     * @date 2022/5/7 19:29
     * @param session
     * @param message
     */
    public static void sendMessage(Session session, String message) throws IOException {
        if(session != null){
            synchronized (session) {
                session.getBasicRemote().sendText(message);
            }
        }
    }

    public static void tryToSend(double totalPrice) {
        if(SessionPools.size()==0){
            return;
        }

        for (Session session: SessionPools.values()) {
            try {
                sendMessage(session, "");
            } catch(Exception e){
                e.printStackTrace();
                continue;
            }
        }
    }
    /**
     * 建立连接成功调用
     * @author 杨捷宁
     * @date 2022/5/7 19:30
     * @param session
     * @param userName 用户名称,用于记录和删除session
     * @param type 需求类型
     */
    @OnOpen
    public void onOpen(Session session, @PathParam(value = "sid") String userName, @PathParam(value = "type") String type){

        SessionPools.put(userName, session);

        addOnlineCount();
        System.out.println("加入UAVHeightWebSocket！当前人数为" + onlineNum);
        try {
            //sendMessage(session, "欢迎" + userName + "加入连接！");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 关闭连接时调用
     * @author 杨捷宁
     * @date 2022/5/7 9:45
     */
    @OnClose
    public void onClose(@PathParam(value = "sid") String userName){

        SessionPools.remove(userName);

        subOnlineCount();
    }

    //收到客户端信息
    @OnMessage
    public void onMessage(String data, Session session, @PathParam(value = "sid") String userName, @PathParam(value = "type") String type) throws IOException{
        try {
            SessionPools.get("3").getBasicRemote().sendText(data);
            SessionPools.get("vue").getBasicRemote().sendText(data);

//            for(ConcurrentHashMap.Entry<String, Session> ce: SessionPools.entrySet()) {
//                ce.getValue().getBasicRemote().sendText(data);
//            }

        } catch (Exception e) {
            System.out.println(e);
        }

    }

    //错误时调用
    @OnError
    public void onError(Session session, Throwable throwable){
        System.out.println("发生错误");
        throwable.printStackTrace();
    }

    public static void addOnlineCount(){
        onlineNum.incrementAndGet();
    }

    public static void subOnlineCount() {
        onlineNum.decrementAndGet();
    }

}
