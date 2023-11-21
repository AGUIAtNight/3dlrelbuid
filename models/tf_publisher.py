
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import tf


def publish_tf():
    print("l 11")
    rospy.init_node('tf_publisher')
    listener = tf.TransformListener()
    broadcaster = tf.TransformBroadcaster()

    rate = rospy.Rate(10)  # 设置发布频率为10Hz

    while not rospy.is_shutdown():
        try:
            print("l 20")
            # 获取链接之间的变换关系
            (trans, rot) = listener.lookupTransform(
                '/base_link', '/left_front_wheel', rospy.Time(0))

            # 发布链接之间的变换关系
            broadcaster.sendTransform(
                trans, rot, rospy.Time.now(), '/left_front_wheel', '/base_link')

            # 添加其他链接之间的变换关系
            (trans2, rot2) = listener.lookupTransform(
                '/base_link', '/right_front_wheel', rospy.Time(0))
            broadcaster.sendTransform(
                trans2, rot2, rospy.Time.now(), '/right_front_wheel', '/base_link')

            # 添加更多链接之间的变换关系...
            print("l36")

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("l 40")
            continue

        rate.sleep()


if __name__ == '__main__':
    try:
        publish_tf()
    except rospy.ROSInterruptException:
        pass
