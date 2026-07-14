import QtQuick 2.15
import QtQuick.Controls 2.15

ApplicationWindow {
    id: mainWindow
    visible: true
    minimumWidth: 1280
    minimumHeight: 800
    width: 1920
    height: 1080
    color: "#1a1a2e"
    title: "Surface AI — Seat AOI Inspection"

    // Title bar
    Rectangle {
        id: titleBar
        anchors { top: parent.top; left: parent.left; right: parent.right }
        height: 48
        color: "#141e30"

        Row {
            anchors { fill: parent; leftMargin: 16; rightMargin: 16 }
            spacing: 16

            Text {
                text: "Surface AI — Seat AOI"
                font { pixelSize: 20; bold: true }
                color: "#e4e6eb"
                anchors.verticalCenter: parent.verticalCenter
            }

            Item { width: 1; height: 1; Layout.fillWidth: true }

            // Pipeline status indicator
            Rectangle {
                width: 8; height: 8; radius: 4
                anchors.verticalCenter: parent.verticalCenter
                color: pipelineVM && pipelineVM.pipelineStatus === "Running" ? "#00e676" : "#757575"
                SequentialAnimation on opacity {
                    running: pipelineVM && pipelineVM.pipelineStatus === "Running"
                    loops: Animation.Infinite
                    NumberAnimation { from: 1.0; to: 0.3; duration: 800 }
                    NumberAnimation { from: 0.3; to: 1.0; duration: 800 }
                }
            }

            Text {
                text: pipelineVM ? pipelineVM.pipelineStatus : "Unknown"
                color: "#a0a8b8"
                font.pixelSize: 14
                anchors.verticalCenter: parent.verticalCenter
            }
        }
    }

    // Tab bar
    TabBar {
        id: tabBar
        anchors { top: titleBar.bottom; left: parent.left; right: parent.right }
        height: 40

        TabButton {
            text: "Monitor"
            cursorShape: Qt.PointingHandCursor
        }
        TabButton {
            text: "History"
            cursorShape: Qt.PointingHandCursor
        }
        TabButton {
            text: "Config"
            cursorShape: Qt.PointingHandCursor
        }
        TabButton {
            text: "Dashboard"
            cursorShape: Qt.PointingHandCursor
        }
    }

    // Content area with crossfade transition
    SwipeView {
        id: swipeView
        anchors {
            top: tabBar.bottom
            left: parent.left; right: parent.right
            bottom: statusBar.top
        }
        currentIndex: tabBar.currentIndex
        interactive: false

        // Crossfade transition on tab change
        popEnter: Transition {
            OpacityAnimator { from: 0.0; to: 1.0; duration: 200; easing.type: Easing.InOutQuad }
        }
        popExit: Transition {
            OpacityAnimator { from: 1.0; to: 0.0; duration: 200; easing.type: Easing.InOutQuad }
        }

        MonitorScreen {}
        HistoryScreen {}
        ConfigScreen {}
        DashboardScreen {}
    }

    // Status bar
    Rectangle {
        id: statusBar
        anchors { left: parent.left; right: parent.right; bottom: parent.bottom }
        height: 32
        color: "#141e30"

        Text {
            anchors { fill: parent; leftMargin: 8; rightMargin: 8 }
            verticalAlignment: Text.AlignVCenter
            text: "Status: " + (pipelineVM ? pipelineVM.pipelineStatus : "—") +
                  "  |  Last frame: #" + (inspectionVM ? inspectionVM.frameId : "—") +
                  "  |  FPS: " + (pipelineVM ? pipelineVM.overallFps.toFixed(1) : "—")
            color: "#a0a8b8"
            font.pixelSize: 12
        }
    }
}
