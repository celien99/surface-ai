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

                ColumnLayout {
                    spacing: 6
                    Layout.fillWidth: true

                    Repeater {
                        model: pipelineVM ? pipelineVM.stageMetrics : []

                        delegate: RowLayout {
                            spacing: 8
                            Layout.fillWidth: true

                            Label {
                                text: modelData.stageType
                                Layout.preferredWidth: 90
                                color: "#a0a8b8"; font.pixelSize: 13
                            }

                            Rectangle {
                                Layout.fillWidth: true
                                height: 6; radius: 3
                                color: "#1f3460"

                                Rectangle {
                                    width: parent.width * Math.min(modelData.avgLatencyMs / 100.0, 1.0)
                                    height: parent.height; radius: 3
                                    color: modelData.avgLatencyMs > 50 ? "#ff1744" : "#448aff"
                                    Behavior on width { NumberAnimation { duration: 200 } }
                                }
                            }

                            Label {
                                text: modelData.avgLatencyMs.toFixed(1) + "ms"
                                Layout.preferredWidth: 55
                                color: "#a0a8b8"
                                font { pixelSize: 12; family: "monospace" }
                            }
                        }
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
