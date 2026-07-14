import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    property int refreshCounter: 0

    RowLayout {
        anchors { fill: parent; margins: 16 }
        spacing: 16

        // Left: Camera preview panel (70%)
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "#0a0a14"
            radius: 12
            border { color: "#1f3460"; width: 2 }

            Image {
                id: cameraPreview
                anchors { fill: parent; margins: 8 }
                fillMode: Image.PreserveAspectFit
                source: "image://pipeline/frame?t=" + refreshCounter
                cache: false
                smooth: true
            }

            // Defect overlays rendered on top of the preview
            // Bound to inspectionVM.defectModel when defects exist
        }

        // Right: Inspection panel (30%)
        Rectangle {
            Layout.preferredWidth: 340
            Layout.fillHeight: true
            color: "#16213e"
            radius: 12

            ColumnLayout {
                anchors { fill: parent; margins: 16 }
                spacing: 16

                // Verdict badge
                VerdictBadge {
                    Layout.alignment: Qt.AlignHCenter
                    verdict: inspectionVM ? inspectionVM.verdict : "—"
                }

                // Severity & Confidence
                ColumnLayout {
                    spacing: 4
                    Label {
                        text: "Severity: " + (inspectionVM ? inspectionVM.severity : "—")
                        color: "#e4e6eb"; font.pixelSize: 16
                    }
                    Label {
                        text: "Confidence: " + (inspectionVM ? Math.round(inspectionVM.confidence * 100) : "—") + "%"
                        color: "#a0a8b8"; font.pixelSize: 14
                    }
                    Label {
                        text: inspectionVM && inspectionVM.recommendation ? inspectionVM.recommendation : "—"
                        color: "#a0a8b8"; font.pixelSize: 13
                        wrapMode: Text.WordWrap; Layout.fillWidth: true
                        maximumLineCount: 2; elide: Text.ElideRight
                    }
                }

                Rectangle { Layout.fillWidth: true; height: 1; color: "#1f3460" }

                // Stage Metrics
                Label {
                    text: "Stage Metrics"
                    color: "#e4e6eb"; font { pixelSize: 16; bold: true }
                }

                Repeater {
                    model: pipelineVM ? pipelineVM.stageMetrics : []

                    delegate: StageMetricsBar {
                        stageName: modelData.stageType
                        latencyMs: modelData.avgLatencyMs
                        maxLatencyMs: 100.0
                        thresholdMs: 50.0
                    }
                }
            }
        }
    }

    // Refresh timer
    Timer {
        interval: 33; repeat: true; running: true
        onTriggered: refreshCounter++
    }
}
