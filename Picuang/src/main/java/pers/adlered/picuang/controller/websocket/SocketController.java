package pers.adlered.picuang.controller.websocket;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.servlet.ModelAndView;

/**
 * @Author: 杨捷宁
 * @DateTime: 2022/5/7 16:20
 * @Description: 该类用于 配置socket请求和转发
 */
@Controller
public class SocketController {
    @Autowired
    private WebSocketServer webSocketServer;
    @Autowired
    private WebSocketServer2 webSocketServer2;
    @Autowired
    private WebSocketServer3 webSocketServer3;
    @Autowired
    private UAVHeightWebSocketServer uAVHeightWebSocketServer;
    @Autowired
    private UAVHeightWebSocketServer2 uAVHeightWebSocketServer2;
    @Autowired
    private WebSocketFireAreaImage webSocketFireAreaImage;
    @Autowired
    private WebSocketCalcResult webSocketCalcResult;
    @RequestMapping("/index")
    public String index() {
        return "index";
    }
    @RequestMapping("/webSocket")
    public ModelAndView socket() {
        ModelAndView mav=new ModelAndView("/webSocket");
        return mav;
    }

    @RequestMapping("/webSocket2")
    public ModelAndView socket2() {
        ModelAndView mav=new ModelAndView("/webSocket2");
        return mav;
    }

    @RequestMapping("/webSocket3")
    public ModelAndView socket3() {
        ModelAndView mav=new ModelAndView("/webSocket3");
        return mav;
    }
    @RequestMapping("/UAVHeightWebSocket")
    public ModelAndView socketUAVHeight() {
        ModelAndView mav=new ModelAndView("/UAVHeightWebSocket");
        return mav;
    }

    @RequestMapping("/UAVHeightWebSocket2")
    public ModelAndView socketUAVHeight2() {
        ModelAndView mav=new ModelAndView("/UAVHeightWebSocket2");
        return mav;
    }

    @RequestMapping("/webSocketFireAreaImage")
    public ModelAndView webSocketFireAreaImageFunc() {
        ModelAndView mav=new ModelAndView("/webSocketFireAreaImage");
        return mav;
    }


    @RequestMapping("/webSocketCalcResult")
    public ModelAndView webSocketCalcResultFunc() {
        ModelAndView mav=new ModelAndView("/webSocketCalcResult");
        return mav;
    }
}
