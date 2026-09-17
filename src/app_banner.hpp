/**
 * @file app_banner.hpp
 *
 * Branded header strip shared by the session setup and settings dialogs.
 *
 * The artwork mirrors the app icon: a cyan coordinate grid on a dark field,
 * bent around a person-shaped lens on the right, with the wordmark on the
 * left. The deflection is the same point-lens form the demo itself uses, so
 * the banner shows the effect rather than illustrating it.
 *
 * This file is part of GravyLensing, a real-time gravitational lensing
 * simulation.
 *
 * GravyLensing is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GravyLensing is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GravyLensing. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <utility>

#include <QColor>
#include <QFont>
#include <QLinearGradient>
#include <QList>
#include <QPainter>
#include <QPainterPath>
#include <QPointF>
#include <QRadialGradient>
#include <QString>
#include <QWidget>

/// Header strip carrying the wordmark and the lensed-grid artwork.
class AppBanner final : public QWidget {
public:
  AppBanner(QString title, QString subtitle, int bannerHeight,
            QWidget *parent = nullptr)
      : QWidget(parent), title_(std::move(title)),
        subtitle_(std::move(subtitle)) {
    setFixedHeight(bannerHeight);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setAccessibleName(title_);
    setAccessibleDescription(subtitle_);
  }

protected:
  void paintEvent(QPaintEvent *) override {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    const QRectF bounds = rect().adjusted(0, 0, -1, -1);
    QPainterPath frame;
    frame.addRoundedRect(bounds, 18, 18);
    painter.setClipPath(frame);

    const QPointF lens(bounds.right() - bounds.height() * 0.78,
                       bounds.center().y());
    paintField(painter, bounds, lens);
    const QPainterPath body = silhouette(bounds, lens);
    painter.setBrush(Qt::NoBrush);
    paintGlow(painter, body, bounds.height() * 0.30, 5);
    paintGrid(painter, bounds, lens);
    paintPerson(painter, bounds, body);
    paintWordmark(painter, bounds);

    painter.setClipping(false);
    painter.setPen(QPen(QColor(255, 255, 255, 28), 1.0));
    painter.setBrush(Qt::NoBrush);
    painter.drawPath(frame);
  }

private:
  /// Near-black sky with a sparse star field, as in the icon.
  static void paintField(QPainter &painter, const QRectF &bounds,
                         const QPointF &lens) {
    QLinearGradient sky(bounds.topLeft(), bounds.bottomRight());
    sky.setColorAt(0.0, QColor("#04050c"));
    sky.setColorAt(1.0, QColor("#0a0b18"));
    painter.setPen(Qt::NoPen);
    painter.setBrush(sky);
    painter.drawRect(bounds);

    QRadialGradient core(lens, bounds.height() * 1.15);
    core.setColorAt(0.0, QColor(255, 137, 65, 60));
    core.setColorAt(1.0, QColor(255, 137, 65, 0));
    painter.setBrush(core);
    painter.drawRect(bounds);

    painter.setBrush(QColor(255, 255, 255, 150));
    quint32 seed = 1979;
    const auto next = [&seed]() {
      seed = seed * 1664525u + 1013904223u;
      return ((seed >> 8) % 1000) / 1000.0;
    };
    for (int star = 0; star < 70; ++star) {
      const QPointF at(bounds.left() + bounds.width() * next(),
                       bounds.top() + bounds.height() * next());
      painter.drawEllipse(at, 0.9, 0.9);
    }
  }

  /// Grid lines drawn through the point-lens deflection, so they bow around
  /// the subject exactly as the demo's background does. Each line is a wide
  /// dim pass under a bright core, which is what gives the icon its neon.
  static void paintGrid(QPainter &painter, const QRectF &bounds,
                        const QPointF &lens) {
    const double height = bounds.height();
    const double spacing = height / 2.1;
    const double strength = spacing * spacing * 1.8;
    const double softening = spacing * 0.5;

    QList<QPainterPath> lines;
    const double step = 4.0;
    for (double x = bounds.left() - spacing * 2; x <= bounds.right() + spacing * 2;
         x += spacing) {
      QPainterPath line;
      for (double y = bounds.top() - spacing * 2;
           y <= bounds.bottom() + spacing * 2; y += step)
        addDeflected(line, QPointF(x, y), lens, strength, softening);
      lines.append(line);
    }
    for (double y = bounds.top() - spacing * 2;
         y <= bounds.bottom() + spacing * 2; y += spacing) {
      QPainterPath line;
      for (double x = bounds.left() - spacing * 2;
           x <= bounds.right() + spacing * 2; x += step)
        addDeflected(line, QPointF(x, y), lens, strength, softening);
      lines.append(line);
    }

    painter.setBrush(Qt::NoBrush);
    for (const auto &pass : {std::pair{height * 0.055, QColor(0, 137, 184, 55)},
                             std::pair{height * 0.022, QColor(0, 160, 214, 130)},
                             std::pair{height * 0.010, QColor(60, 190, 245, 255)}}) {
      painter.setPen(QPen(pass.second, pass.first, Qt::SolidLine, Qt::RoundCap));
      for (const QPainterPath &line : lines)
        painter.drawPath(line);
    }
  }

  /// Push one sample away from the lens and extend the polyline to it.
  static void addDeflected(QPainterPath &line, const QPointF &point,
                           const QPointF &lens, double strength,
                           double softening) {
    const QPointF offset = point - lens;
    const double distance = std::hypot(offset.x(), offset.y()) + softening;
    const QPointF deflected = point + offset * (strength / (distance * distance));
    if (line.elementCount() == 0)
      line.moveTo(deflected);
    else
      line.lineTo(deflected);
  }

  /// Waist-up silhouette, sized from the banner height.
  static QPainterPath silhouette(const QRectF &bounds, const QPointF &lens) {
    const double height = bounds.height();
    const QPointF head(lens.x(), bounds.top() + height * 0.33);
    const double headRadius = height * 0.135;
    const double shoulderY = head.y() + headRadius * 1.5;
    const double halfWidth = height * 0.34;

    QPainterPath body;
    body.addEllipse(head, headRadius, headRadius * 1.08);
    QPainterPath torso;
    torso.moveTo(head.x() - headRadius * 0.55, shoulderY);
    torso.cubicTo(head.x() - halfWidth * 0.92, shoulderY + height * 0.09,
                  head.x() - halfWidth, shoulderY + height * 0.24,
                  head.x() - halfWidth, bounds.bottom() + 2);
    torso.lineTo(head.x() + halfWidth, bounds.bottom() + 2);
    torso.cubicTo(head.x() + halfWidth, shoulderY + height * 0.24,
                  head.x() + halfWidth * 0.92, shoulderY + height * 0.09,
                  head.x() + headRadius * 0.55, shoulderY);
    torso.closeSubpath();
    body.addPath(torso);
    return body.simplified();
  }

  /// Dark core lit from its own edge by the orange glow, ringed by the
  /// icon's cream-to-cyan rim.
  static void paintPerson(QPainter &painter, const QRectF &bounds,
                          const QPainterPath &body) {
    const double height = bounds.height();

    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor("#050610"));
    painter.drawPath(body);

    painter.save();
    painter.setClipPath(body, Qt::IntersectClip);
    painter.setBrush(Qt::NoBrush);
    paintGlow(painter, body, height * 0.17, 10);
    painter.restore();

    QLinearGradient rim(QPointF(0, bounds.top()), QPointF(0, bounds.bottom()));
    rim.setColorAt(0.0, QColor("#ffe9c9"));
    rim.setColorAt(0.45, QColor("#9fe8ff"));
    rim.setColorAt(1.0, QColor("#12b4e8"));
    painter.setPen(QPen(QBrush(rim), std::max(2.0, height * 0.018)));
    painter.setBrush(Qt::NoBrush);
    painter.drawPath(body);
  }

  /// Stack many faint strokes of falling width into a smooth glow: one wide
  /// stroke per step would band, and Qt has no blur without a scene graph.
  static void paintGlow(QPainter &painter, const QPainterPath &path,
                        double maxWidth, int alpha) {
    const int steps = 26;
    for (int step = steps; step >= 1; --step) {
      const double fraction = static_cast<double>(step) / steps;
      QColor glow(255, 137, 65, alpha);
      painter.setPen(QPen(glow, maxWidth * fraction, Qt::SolidLine,
                          Qt::RoundCap, Qt::RoundJoin));
      painter.drawPath(path);
    }
  }

  /// Wordmark and tagline, left aligned away from the subject.
  void paintWordmark(QPainter &painter, const QRectF &bounds) const {
    const double height = bounds.height();
    const double textWidth = bounds.width() - height * 1.45;

    QFont titleFont = font();
    titleFont.setPointSizeF(std::max(17.0, height * 0.185));
    titleFont.setWeight(QFont::Bold);
    titleFont.setLetterSpacing(QFont::AbsoluteSpacing, 1.4);
    painter.setFont(titleFont);
    painter.setPen(Qt::white);
    painter.drawText(QRectF(height * 0.19, height * 0.24, textWidth,
                            height * 0.32),
                     Qt::AlignLeft | Qt::AlignVCenter, title_);

    QFont subtitleFont = font();
    subtitleFont.setPointSizeF(std::max(10.5, height * 0.082));
    subtitleFont.setWeight(QFont::Medium);
    painter.setFont(subtitleFont);
    painter.setPen(QColor("#b9d9e8"));
    painter.drawText(QRectF(height * 0.2, height * 0.58, textWidth,
                            height * 0.2),
                     Qt::AlignLeft | Qt::AlignVCenter, subtitle_);
  }

  QString title_;
  QString subtitle_;
};
