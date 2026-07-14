import QtQuick 2.15

Rectangle {
    property string verdict: "OK"

    width: 160; height: 80
    radius: 12

    color: verdict === "OK" ? Qt.rgba(0, 200/255, 83/255, 0.15) :
           verdict === "NG" ? Qt.rgba(255/255, 23/255, 68/255, 0.15) :
           verdict === "WARN" ? Qt.rgba(255/255, 145/255, 0, 0.15) :
           Qt.rgba(255/255, 193/255, 7/255, 0.15)

    Behavior on color { ColorAnimation { duration: 300 } }

    Text {
        anchors.centerIn: parent
        text: verdict
        font { pixelSize: 48; bold: true }
        color: verdict === "OK" ? "#00c853" :
               verdict === "NG" ? "#ff1744" :
               verdict === "WARN" ? "#ff9100" : "#ffc107"
    }
}
