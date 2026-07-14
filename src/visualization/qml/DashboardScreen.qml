import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    ColumnLayout {
        anchors { fill: parent; margins: 16 }
        spacing: 16

        // KPI cards row
        RowLayout {
            Layout.fillWidth: true
            spacing: 16

            Repeater {
                model: [
                    { label: "Total Frames", value: dashboardVM ? dashboardVM.totalFrames : 0, color: "#448aff" },
                    { label: "OK Rate", value: dashboardVM ? (dashboardVM.okFrames / Math.max(dashboardVM.totalFrames, 1) * 100).toFixed(1) + "%" : "—", color: "#00c853" },
                    { label: "PPM", value: dashboardVM ? dashboardVM.ppm.toFixed(0) : "—", color: "#ff9100" },
                    { label: "NG Count", value: dashboardVM ? dashboardVM.ngFrames : 0, color: "#ff1744" }
                ]

                delegate: Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 100
                    color: "#16213e"
                    radius: 12

                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        hoverEnabled: true
                        onEntered: parent.y = -2
                        onExited: parent.y = 0
                    }

                    Behavior on y { NumberAnimation { duration: 200 } }

                    ColumnLayout {
                        anchors.centerIn: parent
                        spacing: 4

                        Label {
                            text: modelData.value.toString()
                            font { pixelSize: 32; bold: true }
                            color: modelData.color
                            anchors.horizontalCenter: parent.horizontalCenter
                        }
                        Label {
                            text: modelData.label
                            font.pixelSize: 12
                            color: "#a0a8b8"
                            anchors.horizontalCenter: parent.horizontalCenter
                        }
                    }
                }
            }
        }

        // PPM trend chart + defect distribution
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 16

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: "#16213e"
                radius: 12

                ColumnLayout {
                    anchors { fill: parent; margins: 12 }
                    Label {
                        text: "PPM Trend"
                        color: "#e4e6eb"; font { pixelSize: 14; bold: true }
                    }
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        color: "#1a1a2e"
                        radius: 8
                        Label {
                            anchors.centerIn: parent
                            text: "Chart data: " + (dashboardVM ? dashboardVM.ppmTrend.length + " points" : "—")
                            color: "#a0a8b8"
                        }
                    }
                }
            }

            Rectangle {
                Layout.preferredWidth: 300
                Layout.fillHeight: true
                color: "#16213e"
                radius: 12

                ColumnLayout {
                    anchors { fill: parent; margins: 12 }
                    Label {
                        text: "Defect Types"
                        color: "#e4e6eb"; font { pixelSize: 14; bold: true }
                    }
                    ListView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        model: dashboardVM ? dashboardVM.defectTypeModel : null
                        clip: true

                        delegate: RowLayout {
                            width: parent.width - 8
                            Label { text: model.defectName; color: "#e4e6eb" }
                            Item { Layout.fillWidth: true }
                            Label { text: model.count + " (" + model.percentage.toFixed(1) + "%)"; color: "#a0a8b8" }
                        }
                    }
                }
            }
        }
    }
}
