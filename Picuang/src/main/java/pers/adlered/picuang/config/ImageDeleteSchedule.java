package pers.adlered.picuang.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.Scheduled;
import pers.adlered.picuang.tool.ToolBox;

@Configuration  //标记配置类
@EnableScheduling   //开启定时任务
public class ImageDeleteSchedule {
    //添加定时任务
    @Scheduled(cron = "0/60 * * * * ?")
    private void myTasks() {
        if(ToolBox.deleteUploadImages())
            System.out.println("[定时] 删除图片定时任务完成");
        else
            System.out.println("[定时] 无图片需要删除");
    }
}
