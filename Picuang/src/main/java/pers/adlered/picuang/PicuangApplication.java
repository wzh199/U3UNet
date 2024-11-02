package pers.adlered.picuang;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.servlet.ServletContextInitializer;

import javax.servlet.ServletContainerInitializer;
import javax.servlet.ServletContext;
import javax.servlet.ServletException;

@SpringBootApplication
public class PicuangApplication implements ServletContextInitializer {

    public static void main(String[] args) {
        SpringApplication.run(PicuangApplication.class, args);
    }
    @Override
    public void onStartup(ServletContext servletContext) throws ServletException {

        servletContext.setInitParameter("org.apache.tomcat.websocket.textBufferSize","1024000");

    }
}
